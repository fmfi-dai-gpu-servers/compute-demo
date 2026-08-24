"""FMFI DAI compute platform helpers for submitting Ray jobs.

Common usage:

    from compute import run_job

    job_id, status = run_job(
        entrypoint="python train.py",
        pip=["pandas", "scikit-learn"],
    )
"""

from .auth import ensure_token, get_token, is_token_expired
from .client import (
    ComputeClient,
    JobFailedError,
    build_runtime_env,
    load_env,
    run_job,
)
from .provision import ensure_cluster, inventory

__all__ = [
    "ComputeClient",
    "JobFailedError",
    "build_runtime_env",
    "ensure_cluster",
    "inventory",
    "get_token",
    "is_token_expired",
    "load_env",
    "run_job",
]
