# Compute Platform Demos

Practical examples for the GPU-accelerated ML platform at FMFI DAI. Each example is a self-contained project you can clone, configure, and submit to the Ray cluster.

For platform documentation (service URLs, authentication, resource limits), see the [Compute Platform Wiki](https://github.com/fmfi-dai-gpu-servers/compute-wiki).

---

## Examples

| Example | What It Demonstrates |
|---------|---------------------|
| [**CIFAR-10 GPU Marathon**](examples/cifar-gpu-marathon/) | Long-running deep-learning workload: GPU worker pool, population-based hyperparameter evolution, multi-GPU DDP training, live MLflow metrics, model artifacts to S3 |

---

## Quick Start

### 1. Prerequisites

You need a GitHub account that is a **public** member of the [`fmfi-dai-gpu-servers`](https://github.com/orgs/fmfi-dai-gpu-servers) organization. If you haven't used the platform before, follow the [Quick Start](https://github.com/fmfi-dai-gpu-servers/compute-wiki#quick-start) guide in the wiki first.

### 2. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> See [uv docs](https://docs.astral.sh/uv/getting-started/installation/) for alternative install methods (pip, brew, etc.).

### 3. Clone and Install

```bash
git clone https://github.com/fmfi-dai-gpu-servers/compute-demo.git
cd compute-demo
uv sync
```

This creates a `.venv` with all dependencies (Ray, boto3, MLflow, etc.).

### 4. Configure Credentials

Create a `.env` file in the repo root with your credentials:

```bash
# S3 Storage — get these from https://datasets.c.dai.fmph.uniba.sk → "Show Credentials"
S3_URL=https://storage.c.dai.fmph.uniba.sk
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
S3_BUCKET=datasets-yourusername

# MLflow — generate a token at https://mlflow.c.dai.fmph.uniba.sk/oidc/permissions
MLFLOW_TRACKING_URI="https://mlflow.c.dai.fmph.uniba.sk"
MLFLOW_TRACKING_USERNAME="your-username"
MLFLOW_TRACKING_PASSWORD="your-access-token"
```

> `.env` is gitignored — your credentials won't be committed.

### 5. Authenticate with Ray

```bash
eval $(uv run python ray_auth.py)
```

This opens a browser-based device authorization flow and exports `RAY_ADDRESS` and `RAY_AUTH_TOKEN` into your shell. Doing this once up front means submission won't pause to re-authenticate. (Optional — if you skip it, the submission step below triggers the same flow automatically when the token is missing or expired.)

### 6. Run an Example

```bash
uv run python examples/f1-predictor/submit.py
```

This submits the job to the remote Ray cluster and streams logs to your terminal. Run from the repo root — `uv run` uses the `.venv` automatically.

---

## Tutorial: F1 Race Position Predictor

This example trains a model to predict Formula 1 race finishing positions. It exercises all three platform services together:

```
  S3 (data)  ──→  Ray (compute)  ──→  MLflow (tracking)
       ↑                                    │
       └──────── predictions & results ←────┘
```

### What the Pipeline Does

`f1_ray_demo.py` runs a five-stage ML pipeline distributed across the Ray cluster:

1. **Load data from S3 in parallel** — 5 CSV tables loaded simultaneously via `@ray.remote`
2. **Engineer features** — per-driver statistics computed in parallel across hundreds of drivers using `ray.put()` + fan-out
3. **Upload training data** — processed features written back to S3 for workers to download
4. **Distributed hyperparameter search** — 40 random configs + 50 refined configs (90 total) evaluated in parallel with cross-validation
5. **Train final model & log results** — best model retrained on full data, predictions and metrics logged to MLflow

### Step-by-Step Walkthrough

#### Stage 1: Parallel Data Loading from S3

Each table is loaded as an independent Ray task — all 5 downloads run simultaneously:

```python
@ray.remote
def load_csv_from_s3(key: str) -> pd.DataFrame:
    resp = _s3_client().get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(BytesIO(resp["Body"].read()))

tables = ["races", "results", "drivers", "lap_times", "qualifying"]
dfs = ray.get([load_csv_from_s3.remote(f"f1/{t}.csv") for t in tables])
```

**Key pattern:** To read a CSV from S3 into a DataFrame, use `BytesIO` to wrap the response body, then pass it to `pd.read_csv()`.

#### Stage 2: Feature Engineering with ray.put()

After filtering and joining, per-driver features are computed in parallel. The DataFrames are shared efficiently using `ray.put()`:

```python
results_ref = ray.put(results)
lap_avg_ref = ray.put(lap_avg)

futures = [
    build_driver_features.remote(did, results_ref, lap_avg_ref)
    for did in driver_ids
]
features = ray.get(futures)
```

**Key pattern:** `ray.put()` places data in the Ray object store once. All workers reference the same object without re-serialization. Use this whenever multiple tasks read the same large data.

#### Stage 3: Upload Processed Data

The processed training data is written back to S3 so that worker nodes can download it independently:

```python
train_csv = train.to_csv(index=False).encode()
_s3_client().put_object(Bucket=BUCKET, Key="f1/_train_data.csv", Body=train_csv)
```

**Key pattern:** Write intermediate results to S3 so that remote tasks can fetch them without depending on the driver process.

#### Stage 4: Distributed Hyperparameter Search

This is the core of the demo. 90 model configurations are evaluated in parallel across the cluster. Each evaluation is a self-contained `@ray.remote` task:

```python
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

    model = RandomForestRegressor(...) if config["model_type"] == "rf" else GradientBoostingRegressor(...)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    mae = -scores.mean()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("f1-position-predictor")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)
        mlflow.log_metric("mae", mae)

    return {"mae": mae, "config": config}
```

Then all configs are submitted at once:

```python
futures = [evaluate_config.remote(cfg, f"rand-{i}") for i, cfg in enumerate(configs)]
all_results = ray.get(futures)
```

**Key pattern:** Each `@ray.remote` task must be self-contained. Ray tasks may run on different cluster nodes, so you must re-import libraries and re-create clients (S3, MLflow) inside the function. Don't rely on module-level imports or connections.

**Key pattern:** Each task logs its own results to MLflow independently. This means you can monitor the search in real-time — open the MLflow UI and watch runs appear as workers finish.

The search uses two phases:
- **Phase 1 — Random search:** 40 random hyperparameter configurations
- **Phase 2 — Local refinement:** The top 5 configs are perturbed (10 variants each = 50 more) and re-evaluated

#### Stage 5: Final Model & Results

The best configuration is retrained on the full dataset and used to predict the latest race. Results are logged to MLflow as an artifact:

```python
mlflow.log_text(pred_csv, "predictions.csv")
```

### How Job Submission Works

`submit.py` packages your code and sends it to the cluster using the `compute` helper library:

```python
from pathlib import Path

from compute import run_job

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

if __name__ == "__main__":
    job_id, status = run_job(
        entrypoint="python f1_ray_demo.py",
        working_dir=str(HERE),
        env_file=str(ROOT / ".env"),
        pip=["pandas", "scikit-learn", "scipy", "boto3", "python-dotenv", "mlflow"],
    )
    print(f"\nJob {job_id} finished with status: {status}")
```

`run_job` is a single call that authenticates, builds the runtime environment, submits the job, streams its logs to your terminal, and raises if the job fails. Under the hood it assembles the Ray `runtime_env` from three pieces:

| `runtime_env` field | What it does |
|---------------------|--------------|
| `"pip": [...]` | Auto-installs these packages on the worker nodes |
| `"working_dir": "..."` | Ships your local files (the demo scripts) to the workers |
| `"env_vars": {...}` | Forwards your `.env` credentials so the job can reach S3 and MLflow |

Authentication is handled automatically — if `RAY_AUTH_TOKEN` is missing or expired, `run_job` launches the Keycloak device flow from `ray_auth.py`. Logs are then streamed back in real time and the final `JobStatus` is returned.

#### The `compute` library

`run_job` covers the common case in one call. For finer control (submit now, tail later, inspect status, reuse across jobs), use `ComputeClient` directly:

```python
from compute import ComputeClient

client = ComputeClient()                 # auto-resolves address + token
job_id = client.submit(
    entrypoint="python train.py",
    pip=["pandas"],
    env_vars={"EXTRA": "value"},         # added on top of platform creds
)
client.tail_logs(job_id)                 # streams until the job ends
print(client.status(job_id))
```

Reference: `run_job`, `ComputeClient`, `load_env`, `build_runtime_env`, and `JobFailedError` are all importable from `compute`. The platform's service URLs and the credential prefixes forwarded by default (`S3_*`, `MLFLOW_*`) live in `compute/config.py`.

### Monitoring Results

Once the job is running, you can watch progress in two places:

- **Ray Dashboard** at [ray.c.dai.fmph.uniba.sk](https://ray.c.dai.fmph.uniba.sk) — see running tasks, worker utilization, resource usage
- **MLflow UI** at [mlflow.c.dai.fmph.uniba.sk](https://mlflow.c.dai.fmph.uniba.sk) — see experiment runs appearing in real time, compare MAE across configurations, download prediction artifacts

The job also prints direct links when it finishes.

---

## Example: CIFAR-10 GPU Training Marathon

A longer-running, GPU-saturating counterpart to the F1 demo: a single Ray job that keeps every GPU worker busy for a configurable wall-clock budget (default ~2 hours, scalable to any duration).

```
  S3 (CIFAR-10) ──→  GPU worker pool (Ray actors, autoscaled from zero)
                          │
                          ├─ Stage 1: HPO evolution — population-based search,
                          │            every trial logs per-epoch metrics to MLflow
                          │
                          └─ Stage 2: final multi-GPU DDP training (gloo) of the
                                       best config → weights uploaded to S3
```

### Run It

```bash
# one-time: upload CIFAR-10 to your S3 bucket (~170 MB)
uv run python examples/cifar-gpu-marathon/upload_dataset.py

# default: ~2 h budget, 4 GPU workers, 3-minute HPO trials
uv run python examples/cifar-gpu-marathon/submit.py
```

Scale it without touching code — set any of these before `submit.py`:

| Variable | Default | Effect |
|----------|---------|--------|
| `MARATHON_RUNTIME_MIN` | `120` | Total job budget; HPO gets ~60%, final DDP ~25% |
| `MARATHON_NUM_GPUS` | `4` | Parallel trial workers (capped at cluster capacity) |
| `MARATHON_TRIAL_MIN` | `3` | Minutes per HPO trial — longer trials converge further |

For example, a 6-hour soak test with 5-minute trials:

```bash
MARATHON_RUNTIME_MIN=360 MARATHON_TRIAL_MIN=5 uv run python examples/cifar-gpu-marathon/submit.py
```

### What Makes It Different from the F1 Demo

- **GPU actors instead of CPU tasks.** Each `TrialWorker` actor reserves one HAMi vGPU via `@ray.remote(num_gpus=1)`, downloads CIFAR-10 from S3 once into RAM, and then serves many HPO trials without re-loading data. Creating the actors triggers the Ray autoscaler's scale-from-zero; actors stay alive for the whole job so workers never hit the 120 s idle timeout mid-run.
- **Population-based evolution.** Random configs first; once results exist, the top-4 configs breed mutated children (width, depth, LR, batch size, label smoothing). Trial length is bounded by a deadline so the stage lands on budget regardless of epoch speed.
- **Hand-rolled multi-GPU DDP.** The final stage converts the same actors into `torch.distributed` ranks (gloo backend over TCP, master = worker 0's pod IP) — no NVLink assumptions, works across vGPU slices and nodes.
- **Live MLflow.** Every trial is an MLflow run streaming per-epoch `val_acc`/`train_loss`; the job closes with a `marathon-summary` run logging the best config and final DDP accuracy.

### Cluster Gotchas Baked Into the Code

- **The head node has no torch.** The job driver must never `import torch`, and every actor return value must be plain JSON primitives. Returning e.g. `torch.__version__` (a `TorchVersion` str-subclass) embeds a torch-module reference in the pickle — unpickling on the head fails with `ModuleNotFoundError` and kills the job. Catch exceptions inside GPU code and return `str(...)` summaries.
- **vGPU memory is capped (~2 GiB per worker).** The config space constrains `width` against `batch` (measured: batch 256 / width 64 ≈ 1.05 GiB peak) and OOM'd trials are reported and skipped, not fatal.
- **Workers are 1.5 CPU / 3.5 GiB RAM.** Data augmentation stays in numpy, evaluation batches are 512.
- **MLflow artifact uploads currently fail** with `PermissionError` from job pods (`log_text`/`log_artifact`); metrics and params log fine. The marathon therefore stores model weights in S3 and logs metrics to MLflow; the summary is printed as `RESULT_JSON` on stdout instead of an artifact.

---


## GPU Allocation on the Platform

Every user gets their **own RayCluster** with their **own Volcano queue** — no
more competing inside one shared cluster. This happens automatically:
`ComputeClient` (and therefore `run_job`) calls the platform provisioner
(`provision.c.dai.fmph.uniba.sk`) with your Keycloak token before submitting,
which creates on first use:

- a Volcano queue `u-<username>` with a fair-share **weight** (relative share
  when GPUs are contended) and a hard **capability** ceiling (default: 2 GPU
  workers + your head pod);
- a RayCluster `ray-<username>` (workers scale 0 → max inside your queue) at
  `ray-<username>.c.dai.fmph.uniba.sk` behind the same Keycloak login;
- an idle garbage collector that deletes clusters unused for 24 h (recreated
  transparently on your next submit).

Semantics: idle capacity is **borrowable** up to your capability ceiling;
preemption is off, so borrowed GPUs are only released when the borrowing job
finishes. The shared cluster remains available with `RAY_SHARED=1` (or
`ComputeClient(user_cluster=False)`).

Admin knobs (ConfigMaps in `ray-system`, no redeploys):

- `ray-user-quotas` — `quotas.yaml` with `defaults` plus per-`users:` overrides
  (`weight`, `capability.{cpu,memory,volcano.sh/vgpu-number}`, `max_workers`,
  `idle_hours`) and `groups:` weights;
- `ray-user-groups` — one `username: group` line per user; grouped users get
  their queue placed under `g-<group>`, so a group shares its parent slice.

Note: the Keycloak gate is per-org, not per-cluster — any org member who knows
your cluster URL can open your dashboard. The CLI always targets your own
cluster.

## Key Patterns Reference

### Self-Contained Remote Tasks

Ray tasks run on arbitrary cluster nodes. Every `@ray.remote` function must import its own dependencies and create its own clients:

```python
@ray.remote
def my_task(data_id: int):
    import boto3, mlflow, pandas as pd

    s3 = boto3.client("s3", endpoint_url=os.getenv("S3_URL"), ...)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    # ... work ...
```

Do **not** rely on module-level imports or global client objects inside `@ray.remote` functions.

### Sharing Large Data with ray.put()

When multiple tasks read the same data, place it in the object store once:

```python
data_ref = ray.put(large_dataframe)
futures = [process.remote(data_ref, i) for i in range(100)]
results = ray.get(futures)
```

Without `ray.put()`, the data would be serialized separately for each task.

### Configuring with .env + dotenv

Centralize all service credentials in a root `.env` file. Two layers use it:

**Inside your job script** (runs on the cluster), load it with `python-dotenv` so local `os.getenv(...)` calls resolve:

```python
from dotenv import load_dotenv
load_dotenv()
```

**At submission time**, `run_job` / `ComputeClient.submit` read the `.env` automatically and forward the platform credentials (`S3_*`, `MLFLOW_*`) to workers via `runtime_env["env_vars"]` — so workers can reach S3 and MLflow without needing the `.env` file. Pass `env_file=` to point elsewhere, or `env_vars={...}` to add extra variables on top.

### Reading DataFrames from S3

```python
from io import BytesIO
import pandas as pd
import boto3

s3 = boto3.client("s3", endpoint_url="https://storage.c.dai.fmph.uniba.sk", ...)
resp = s3.get_object(Bucket="datasets-username", Key="path/to/data.csv")
df = pd.read_csv(BytesIO(resp["Body"].read()))
```

### Writing DataFrames to S3

```python
csv_bytes = df.to_csv(index=False).encode()
s3.put_object(Bucket="datasets-username", Key="path/to/output.csv", Body=csv_bytes)
```

### MLflow from Ray Tasks

Each remote task creates its own MLflow connection and logs independently:

```python
@ray.remote
def train_and_log(config: dict):
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("my-experiment")
    with mlflow.start_run(run_name="worker-run"):
        mlflow.log_params(config)
        mlflow.log_metric("accuracy", 0.95)
```

This works because MLflow's server tracks runs centrally — multiple workers can log concurrently without conflicts.

### Memory Management on Shared Clusters

The cluster has limited resources. Free references to large objects between pipeline stages:

```python
del large_dataframe, ray_object_ref
import gc
gc.collect()
```

This releases memory in the Ray object store before the next compute-heavy stage.
