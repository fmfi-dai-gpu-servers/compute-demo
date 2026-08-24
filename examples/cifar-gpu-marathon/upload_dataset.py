"""Upload CIFAR-10 to the platform S3 bucket (run once, from your machine).

Downloads CIFAR-10 from the Hugging Face mirror (the original
cs.toronto.edu server is often very slow) and re-packages it into the
classic ``cifar-10-batches-py`` pickle layout, uploaded to S3 under
``cifar10/`` so cluster workers can fetch it without external internet
access. Already-uploaded batches are skipped.
"""

import io
import os
import pickle

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

PREFIX = "cifar10"
TRAIN_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet"
TEST_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"
LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]


def load_xy(parquet_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path)
    x = np.zeros((len(df), 3, 32, 32), dtype=np.uint8)
    y = np.zeros(len(df), dtype=np.int64)
    img_col = "img" if "img" in df.columns else "image"
    for i, (img_rec, label) in enumerate(zip(df[img_col], df["label"])):
        png = img_rec["bytes"] if isinstance(img_rec, dict) else img_rec
        im = Image.open(io.BytesIO(png)).convert("RGB")
        x[i] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
        y[i] = label
    return x, y


def main() -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_URL"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        region_name="us-east-1",
    )
    bucket = os.getenv("S3_BUCKET")

    existing = {
        obj["Key"]
        for obj in s3.list_objects_v2(Bucket=bucket, Prefix=PREFIX).get("Contents", [])
    }
    wanted = [f"{PREFIX}/data_batch_{i}" for i in range(1, 6)] + [
        f"{PREFIX}/test_batch", f"{PREFIX}/batches.meta"
    ]
    missing = [k for k in wanted if k not in existing]
    if not missing:
        print(f"CIFAR-10 already present in s3://{bucket}/{PREFIX}/ — nothing to do.")
        return

    def put(key: str, body: bytes) -> None:
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        print(f"  uploaded s3://{bucket}/{key} ({len(body)} bytes)")

    if any(k.startswith(f"{PREFIX}/data_batch") for k in missing):
        print(f"Downloading {TRAIN_URL} ...")
        x, y = load_xy(TRAIN_URL)
        order = np.random.default_rng(42).permutation(len(x))
        per = len(x) // 5
        for b in range(5):
            sl = order[b * per : (b + 1) * per]
            body = pickle.dumps(
                {"data": x[sl].reshape(len(sl), -1), "labels": y[sl].tolist()}, protocol=2
            )
            put(f"{PREFIX}/data_batch_{b + 1}", body)
    if f"{PREFIX}/test_batch" in missing:
        print(f"Downloading {TEST_URL} ...")
        x, y = load_xy(TEST_URL)
        body = pickle.dumps({"data": x.reshape(len(x), -1), "labels": y.tolist()}, protocol=2)
        put(f"{PREFIX}/test_batch", body)
    if f"{PREFIX}/batches.meta" in missing:
        put(f"{PREFIX}/batches.meta", pickle.dumps({"label_names": LABELS}, protocol=2))

    print("Done.")


if __name__ == "__main__":
    main()
