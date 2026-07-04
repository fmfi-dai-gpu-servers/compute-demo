"""High-level client for submitting jobs to the FMFI DAI Ray cluster.

Wraps :class:`ray.job_submission.JobSubmissionClient` with the platform's
auth, service URLs, and credential-forwarding conventions baked in, so most
jobs go from ~40 lines of boilerplate to a single :func:`run_job` call.

    from compute import run_job

    job_id, status = run_job(
        entrypoint="python train.py",
        pip=["pandas", "scikit-learn"],
    )

For finer control (submit now, tail later, reuse across jobs) use
:class:`ComputeClient` directly.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from ray.job_submission import JobStatus, JobSubmissionClient

from . import auth, config

# JobStatus values that will not change on their own.
_TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}
)


class JobFailedError(RuntimeError):
    """Raised by :func:`run_job` when a job ends in a non-success terminal state."""

    def __init__(self, job_id: str, status: JobStatus):
        self.job_id = job_id
        self.status = status
        super().__init__(f"Job {job_id} ended with status {status}")


def _strip_quotes(value: str) -> str:
    """Remove a single layer of surrounding matching quotes."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _is_platform_key(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in config.PLATFORM_ENV_PREFIXES)


def load_env(path: str | os.PathLike[str] = ".env") -> dict[str, str]:
    """Load variables from a dotenv file, stripping surrounding quotes.

    Mirrors the previous submit.py behaviour (``dotenv_values`` +
    ``v.strip('"')``) so quoted values such as ``KEY="value"`` come through
    clean. Missing file → empty dict.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    return {k: _strip_quotes(v) for k, v in dotenv_values(p).items()}


def build_runtime_env(
    *,
    pip: list[str] | str | None = None,
    working_dir: str | None = ".",
    env_vars: dict[str, str] | None = None,
    conda: dict[str, Any] | str | None = None,
    py_modules: list[str] | None = None,
    excludes: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble a Ray ``runtime_env`` dict, omitting unset fields."""
    env: dict[str, Any] = {}
    if working_dir is not None:
        env["working_dir"] = working_dir
    if pip is not None:
        env["pip"] = pip
    if conda is not None:
        env["conda"] = conda
    if py_modules is not None:
        env["py_modules"] = py_modules
    if excludes is not None:
        env["excludes"] = excludes
    if env_vars:
        env["env_vars"] = env_vars
    return env


class ComputeClient:
    """Authenticated client for the FMFI DAI Ray cluster.

    Address and token resolve from (1) constructor args, then (2) the
    ``RAY_ADDRESS`` / ``RAY_AUTH_TOKEN`` environment variables, then (3) the
    platform defaults from :mod:`compute.config`. If no usable token is found
    the Keycloak device flow is triggered automatically.
    """

    def __init__(self, address: str | None = None, token: str | None = None):
        resolved_address = address or os.environ.get(
            auth.ENV_ADDRESS, config.RAY_ADDRESS
        )
        resolved_token = token or auth.ensure_token()
        self.address = resolved_address
        self._client = JobSubmissionClient(
            address=resolved_address,
            headers={"Authorization": f"Bearer {resolved_token}"},
        )

    def submit(
        self,
        entrypoint: str,
        *,
        pip: list[str] | str | None = None,
        working_dir: str | None = ".",
        env_file: str | None = ".env",
        env_vars: dict[str, str] | None = None,
        forward_all_env: bool = False,
        runtime_env: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a job and return its job id.

        By default the platform credentials (``S3_*``, ``MLFLOW_*``) are loaded
        from ``env_file`` and forwarded to the workers; pass additional vars via
        ``env_vars``. Give ``runtime_env=`` to bypass the builder entirely.
        Extra keyword args (e.g. ``job_id=``, ``metadata=``) are forwarded to
        :meth:`JobSubmissionClient.submit_job`.
        """
        if runtime_env is None:
            forwarded: dict[str, str] = {}
            if env_file:
                loaded = load_env(env_file)
                forwarded.update(
                    loaded
                    if forward_all_env
                    else {k: v for k, v in loaded.items() if _is_platform_key(k)}
                )
            if env_vars:
                forwarded.update(env_vars)
            runtime_env = build_runtime_env(
                pip=pip,
                working_dir=working_dir,
                env_vars=forwarded or None,
            )
        return self._client.submit_job(
            entrypoint=entrypoint, runtime_env=runtime_env, **kwargs
        )

    def tail_logs(self, job_id: str, *, file=None) -> None:
        """Stream a job's logs to ``file`` (stdout) until the job terminates."""
        out = file if file is not None else sys.stdout

        async def _tail() -> None:
            async for line in self._client.tail_job_logs(job_id):
                print(line, end="", file=out)

        asyncio.run(_tail())

    def status(self, job_id: str) -> JobStatus:
        return self._client.get_job_status(job_id)

    def logs(self, job_id: str) -> str:
        return self._client.get_job_logs(job_id)

    def list_jobs(self) -> list:
        return self._client.list_jobs()

    def stop(self, job_id: str) -> None:
        self._client.stop_job(job_id)

    def delete(self, job_id: str) -> None:
        self._client.delete_job(job_id)

    def wait(self, job_id: str, *, poll_interval: float = 5.0) -> JobStatus:
        """Poll until the job reaches a terminal state and return its status."""
        status = self.status(job_id)
        while status not in _TERMINAL_STATUSES:
            time.sleep(poll_interval)
            status = self.status(job_id)
        return status


def run_job(
    entrypoint: str,
    *,
    pip: list[str] | str | None = None,
    working_dir: str | None = ".",
    env_file: str | None = ".env",
    env_vars: dict[str, str] | None = None,
    forward_all_env: bool = False,
    runtime_env: dict[str, Any] | None = None,
    tail: bool = True,
    client: ComputeClient | None = None,
    **kwargs: Any,
) -> tuple[str, JobStatus]:
    """Submit a job, stream its logs, and return ``(job_id, status)``.

    Accepts the same arguments as :meth:`ComputeClient.submit`. Logs are
    streamed to stdout (disabled with ``tail=False``, in which case the job is
    polled until completion). Raises :class:`JobFailedError` if the job ends
    STOPPED, FAILED, or TERMINATED.
    """
    c = client or ComputeClient()
    job_id = c.submit(
        entrypoint,
        pip=pip,
        working_dir=working_dir,
        env_file=env_file,
        env_vars=env_vars,
        forward_all_env=forward_all_env,
        runtime_env=runtime_env,
        **kwargs,
    )
    print(f"Submitted job: {job_id}\n", file=sys.stderr)

    if tail:
        c.tail_logs(job_id)
    else:
        c.wait(job_id)

    status = c.status(job_id)
    print(f"\nJob {job_id} finished with status: {status}", file=sys.stderr)
    if status not in (JobStatus.SUCCEEDED,):
        raise JobFailedError(job_id, status)
    return job_id, status
