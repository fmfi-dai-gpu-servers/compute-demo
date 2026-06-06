import json
import os
import time
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()

import boto3
import pandas as pd
import ray

ray.init()

print(f"Ray cluster resources: {ray.cluster_resources()}")
print(f"Available GPUs: {ray.available_resources().get('GPU', 0)}")
print(f"Nodes: {len(ray.nodes())}")

S3_URL = os.getenv("S3_URL", "https://storage.c.dai.fmph.uniba.sk")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
BUCKET = os.getenv("S3_BUCKET")


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="us-east-1",
    )


@ray.remote
def load_csv_from_s3(key: str) -> pd.DataFrame:
    resp = _s3_client().get_object(Bucket=BUCKET, Key=key)
    df = pd.read_csv(BytesIO(resp["Body"].read()))
    print(f"  Loaded {key}: {len(df)} rows")
    return df


tables = ["races", "results", "drivers", "lap_times", "qualifying"]
print(f"\nLoading {len(tables)} tables from S3 in parallel...")
t0 = time.time()
dfs = ray.get([load_csv_from_s3.remote(f"f1/{t}.csv") for t in tables])
print(f"Loaded all tables in {time.time() - t0:.1f}s")

races, results, drivers, lap_times, qualifying = dfs

races_2000 = races[races["year"] >= 2000]
valid_race_ids = set(races_2000["raceId"])
results = results[results["raceId"].isin(valid_race_ids)]
lap_times = lap_times[lap_times["raceId"].isin(valid_race_ids)]
print(f"\nFiltered to year >= 2000: {len(results)} results, {len(lap_times)} lap times")

lap_avg = lap_times.groupby("driverId")["milliseconds"].mean().reset_index()
lap_avg.columns = ["driverId", "avg_lap_ms"]


@ray.remote
def build_driver_features(
    driver_id: int, results_df, lap_avg_df
) -> dict | None:
    dr = results_df[results_df["driverId"] == driver_id]
    if dr.empty:
        return None
    avg_grid = dr["grid"].mean()
    avg_finish = dr["positionOrder"].mean()
    avg_points = dr["points"].mean()
    dnfs = (dr["positionText"] == "R").sum()
    total_races = len(dr)
    lap_row = lap_avg_df[lap_avg_df["driverId"] == driver_id]
    avg_lap_ms = lap_row["avg_lap_ms"].iloc[0] if not lap_row.empty else 0
    return {
        "driverId": driver_id,
        "avg_grid_pos": avg_grid,
        "avg_finish_pos": avg_finish,
        "avg_points": avg_points,
        "dnf_rate": dnfs / max(total_races, 1),
        "total_races": total_races,
        "avg_lap_ms": avg_lap_ms,
    }


print("\nEngineering features for all drivers in parallel...")
t0 = time.time()
driver_ids = results["driverId"].unique()

results_ref = ray.put(results)
lap_avg_ref = ray.put(lap_avg)

futures = [
    build_driver_features.remote(did, results_ref, lap_avg_ref)
    for did in driver_ids
]
features = ray.get(futures)
features = [f for f in features if f is not None]
driver_feats = pd.DataFrame(features)
print(f"Built features for {len(driver_feats)} drivers in {time.time() - t0:.1f}s")

train = results.merge(driver_feats, on="driverId")
train = train.merge(races_2000[["raceId", "year"]], on="raceId")
train = train[train["positionText"] != "R"]
train = train[~train["positionOrder"].isna()]
train["target"] = train["positionOrder"].astype(float)

feature_cols = [
    "grid",
    "avg_grid_pos",
    "avg_finish_pos",
    "avg_points",
    "dnf_rate",
    "total_races",
    "avg_lap_ms",
    "year",
]
train = train.dropna(subset=feature_cols + ["target"])
print(f"\nTraining set: {len(train)} samples, {len(feature_cols)} features")

train_data_key = "f1/_train_data.csv"
train_csv = train.to_csv(index=False).encode()
_s3_client().put_object(Bucket=BUCKET, Key=train_data_key, Body=train_csv)
print(f"Uploaded training data to S3 ({len(train_csv)} bytes)")

drivers_key = "f1/_drivers.csv"
drivers_csv = drivers[["driverId", "surname"]].to_csv(index=False).encode()
_s3_client().put_object(Bucket=BUCKET, Key=drivers_key, Body=drivers_csv)

import gc

del dfs, races, lap_times, qualifying, results, driver_feats, lap_avg
del results_ref, lap_avg_ref, train_csv, drivers_csv
gc.collect()


@ray.remote
def evaluate_config(config: dict, run_name: str) -> dict:
    import boto3
    import mlflow
    import pandas as pd
    from io import BytesIO
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_URL"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        region_name="us-east-1",
    )
    resp = s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key="f1/_train_data.csv")
    train = pd.read_csv(BytesIO(resp["Body"].read()))

    feature_cols = [
        "grid", "avg_grid_pos", "avg_finish_pos", "avg_points",
        "dnf_rate", "total_races", "avg_lap_ms", "year",
    ]
    X = train[feature_cols].values
    y = train["target"].values

    if config["model_type"] == "rf":
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=42,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=42,
        )
    scores = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_absolute_error"
    )
    mae = -scores.mean()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("f1-position-predictor")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in config.items()})
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("training_samples", len(train))

    print(f"  {config['model_type']} n={config['n_estimators']} -> MAE={mae:.3f} (logged to MLflow)")
    return {"mae": mae, "config": config}


import numpy as np
from scipy.stats import loguniform

N_RANDOM = 40
N_TOP_REFINE = 10
REFINE_STD = {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.3, "min_samples_split": 1}

rng = np.random.RandomState(42)
configs = []

model_types = ["rf", "gb"]
for _ in range(N_RANDOM):
    mt = rng.choice(model_types)
    cfg = {
        "model_type": mt,
        "n_estimators": int(rng.choice([50, 100, 150, 200, 300, 500])),
        "max_depth": rng.choice([3, 4, 5, 7, 10, 15, 20, 30, 50, None]),
        "learning_rate": float(loguniform.rvs(1e-3, 1e-1, random_state=rng)),
    }
    if mt == "rf":
        cfg["min_samples_split"] = int(rng.choice(range(2, 20)))
        cfg["max_features"] = rng.choice(["sqrt", "log2", None])
    else:
        cfg["subsample"] = float(rng.uniform(0.6, 1.0))
        cfg["min_samples_leaf"] = int(rng.choice(range(1, 10)))
    configs.append(cfg)

print(f"\nPhase 1: Random search ({N_RANDOM} configs)...")
print("Watch the Ray Dashboard: https://ray.c.dai.fmph.uniba.sk\n")

t0 = time.time()
futures = [evaluate_config.remote(cfg, f"rand-{i:02d}-{cfg['model_type']}") for i, cfg in enumerate(configs)]
all_results = ray.get(futures)

all_results.sort(key=lambda r: r["mae"])
top_results = all_results[:5]
print(f"\nPhase 1 completed in {time.time() - t0:.1f}s")
print(f"Top 5 MAEs: {[round(r['mae'], 3) for r in top_results]}")

refine_configs = []
for base in top_results:
    for _ in range(N_TOP_REFINE):
        cfg = dict(base["config"])
        cfg["n_estimators"] = max(10, int(rng.normal(cfg.get("n_estimators", 100), REFINE_STD["n_estimators"])))
        if isinstance(cfg.get("max_depth"), int):
            cfg["max_depth"] = max(1, int(rng.normal(cfg["max_depth"], REFINE_STD["max_depth"])))
        if "learning_rate" in cfg:
            lr = cfg["learning_rate"] * float(np.exp(rng.normal(0, REFINE_STD["learning_rate"])))
            cfg["learning_rate"] = max(1e-4, min(0.5, lr))
        if "min_samples_split" in cfg:
            cfg["min_samples_split"] = max(2, int(rng.normal(cfg["min_samples_split"], REFINE_STD["min_samples_split"])))
        refine_configs.append(cfg)

print(f"\nPhase 2: Refining top configs ({len(refine_configs)} configs)...")
t1 = time.time()
futures = [evaluate_config.remote(cfg, f"refine-{i:02d}-{cfg['model_type']}") for i, cfg in enumerate(refine_configs)]
refine_results = ray.get(futures)

all_results.extend(refine_results)
best_result = min(all_results, key=lambda r: r["mae"])
best_cfg = best_result["config"]
print(f"\nPhase 2 completed in {time.time() - t1:.1f}s")
print(f"Total tuning time: {time.time() - t0:.1f}s")
print(f"Best config: {best_cfg}")
print(f"Best MAE: {best_result['mae']:.3f} positions")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

X_train = train[feature_cols].values
y_train = train["target"].values

if best_cfg["model_type"] == "rf":
    final_model = RandomForestRegressor(
        n_estimators=best_cfg["n_estimators"],
        max_depth=best_cfg["max_depth"],
        min_samples_split=best_cfg["min_samples_split"],
        random_state=42,
    )
else:
    final_model = GradientBoostingRegressor(
        n_estimators=best_cfg["n_estimators"],
        max_depth=best_cfg["max_depth"],
        learning_rate=best_cfg["learning_rate"],
        random_state=42,
    )

final_model.fit(X_train, y_train)

latest = train[train["year"] == train["year"].max()].head(10)
latest = latest.merge(drivers[["driverId", "surname"]], on="driverId")
preds = final_model.predict(latest[feature_cols].values)
latest["predicted_pos"] = preds.round(1)

print("\n--- Predictions for latest race ---")
print(latest[["surname", "grid", "target", "predicted_pos"]].to_string(index=False))

output = {
    "best_config": {k: str(v) for k, v in best_cfg.items()},
    "mae": best_result["mae"],
    "training_samples": len(train),
    "predictions": latest[["surname", "grid", "target", "predicted_pos"]]
    .round(2)
    .to_dict("records"),
}
print(f"\nRESULT_JSON:{json.dumps(output)}")

import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("f1-position-predictor")
with mlflow.start_run(run_name="f1-best-model"):
    mlflow.log_params({k: str(v) for k, v in best_cfg.items()})
    mlflow.log_metric("mae", best_result["mae"])
    mlflow.log_metric("training_samples", len(train))
    pred_csv = latest[["surname", "grid", "target", "predicted_pos"]].to_csv(index=False)
    mlflow.log_text(pred_csv, "predictions.csv")
    mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "").strip('"')
    print(f"\nLogged best model to MLflow: {mlflow.active_run().info.run_id}")
    print(f"  -> {mlflow_url}")

print(f"\n{'=' * 50}")
print(f"Ray Dashboard:  https://ray.c.dai.fmph.uniba.sk")
print(f"MLflow UI:      https://mlflow.c.dai.fmph.uniba.sk")
print(f"Datasets:       https://datasets.c.dai.fmph.uniba.sk")
print(f"{'=' * 50}")

ray.shutdown()
