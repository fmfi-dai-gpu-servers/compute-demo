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
            # the pip list transitively upgrades cryptography past what the
            # base image's pyOpenSSL 24.2.1 can bind to (OpenSSL._lib.GEN_EMAIL
            # AttributeError) — ship a matching pyOpenSSL in the job venv
            "pyopenssl>=25",
        ],
    )
    print(f"\nJob {job_id} finished with status: {status}")
