"""Submit the CIFAR-10 GPU Training Marathon to the Ray cluster.

Environment overrides (no code changes needed):

    MARATHON_RUNTIME_MIN=240   # 4-hour run
    MARATHON_NUM_GPUS=4        # use every GPU worker
    MARATHON_TRIAL_MIN=5       # longer HPO trials
"""

import os

from pathlib import Path

from compute import run_job

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

if __name__ == "__main__":
    marathon_env = {
        k: os.environ[k]
        for k in ("MARATHON_RUNTIME_MIN", "MARATHON_NUM_GPUS", "MARATHON_TRIAL_MIN")
        if k in os.environ
    }
    job_id, status = run_job(
        entrypoint="python gpu_marathon.py",
        working_dir=str(HERE),
        env_file=str(ROOT / ".env"),
        env_vars=marathon_env,
        pip=["mlflow", "boto3", "python-dotenv"],
    )
    print(f"\nJob {job_id} finished with status: {status}")
