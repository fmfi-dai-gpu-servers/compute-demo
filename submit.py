import asyncio
import json
import os

from dotenv import dotenv_values
from ray.job_submission import JobSubmissionClient

env = dotenv_values(".env")

client = JobSubmissionClient(
    address="https://ray.c.dai.fmph.uniba.sk",
    headers={"Authorization": f"Bearer {os.environ['RAY_AUTH_TOKEN']}"},
)

runtime_env = {
    "pip": [
        "pandas",
        "scikit-learn",
        "scipy",
        "boto3",
        "python-dotenv",
        "mlflow",
    ],
    "working_dir": ".",
    "env_vars": {k: v.strip('"') for k, v in env.items()},
}

job_id = client.submit_job(
    entrypoint="python f1_ray_demo.py",
    runtime_env=runtime_env,
)

print(f"Submitted job: {job_id}\n")


async def tail_logs():
    async for line in client.tail_job_logs(job_id):
        print(line, end="")


asyncio.run(tail_logs())

status = client.get_job_status(job_id)
print(f"\nJob {job_id} finished with status: {status}")
