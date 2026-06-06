import io
import os
import zipfile
from pathlib import Path

import boto3
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_URL"),
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    region_name="us-east-1",
)

BUCKET = os.getenv("S3_BUCKET")


def main():
    s3.download_file(BUCKET, "f1/races.csv", "races.csv")

    # with zipfile.ZipFile("archive.zip") as zf:
    #     for name in zf.namelist():
    #         if not name.endswith(".csv"):
    #             continue
    #         csv_data = zf.read(name)
    #         key = f"f1/{os.path.basename(name)}"
    #         s3.put_object(Bucket=BUCKET, Key=key, Body=csv_data)
    #         print(f"uploaded {key}")


if __name__ == "__main__":
    main()
