from pathlib import Path

from compute import run_job

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

if __name__ == "__main__":
    job_id, status = run_job(
        entrypoint="python f1_ray_demo.py",
        working_dir=str(HERE),
        env_file=str(ROOT / ".env"),
        pip=[
            "pandas",
            "scikit-learn",
            "scipy",
            "boto3",
            "python-dotenv",
            "mlflow",
        ],
    )
    print(f"\nJob {job_id} finished with status: {status}")
