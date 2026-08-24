"""Provision (or attach to) the calling user's dedicated Ray cluster.

Every platform user gets their own RayCluster inside their own Volcano queue
(the server side is the ``ray-provisioner`` service in the ``services``
repository). This module calls the platform provisioner with the user's
Keycloak token and returns the address of their cluster, creating it on
first use.

Users may declare exact worker resources::

    from compute import ensure_cluster, run_job

    ensure_cluster(resources={"cpus": 4, "mem_gb": 12, "gpus": 1,
                              "gpu_frac": 0.3, "gpu_type": "titan v"})

    run_job("python train.py",
            resources={"gpu_frac": 1.0, "gpu_type": "TITAN Xp"})

    python -m compute.provision   # list available GPU models
"""

from __future__ import annotations

import requests

from . import auth, config

RESOURCES_KEYS = ("cpus", "mem_gb", "gpus", "gpu_frac", "gpu_mem_gb", "gpu_type")


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token or auth.ensure_token()}"}


def ensure_cluster(token: str | None = None, resources: dict | None = None) -> str:
    """Return the address of the caller's dedicated Ray cluster.

    Calls ``/ensure`` with the Keycloak bearer token; the provisioner creates
    the Volcano queue, RayCluster and ingress on first use and waits until the
    cluster is ready. ``resources`` (all keys optional) declares the worker
    shape: cpus, mem_gb, gpus, gpu_frac (fraction of a card, default 0.2) or
    gpu_mem_gb (exact), and gpu_type (model pin, see :func:`inventory`).
    Priority is derived from the request size by the platform, not settable.
    """
    payload = None
    if resources:
        unknown = set(resources) - set(RESOURCES_KEYS)
        if unknown:
            raise ValueError(f"unknown resource keys {sorted(unknown)}; valid: {RESOURCES_KEYS}")
        payload = resources
    last_error = "provisioner unreachable"
    resp = None
    for _ in range(2):  # one retry on transport/5xx
        try:
            resp = requests.get(
                f"{config.PROVISIONER_URL}/ensure",
                headers=_headers(token),
                json=payload,
                timeout=180,  # /ensure waits for cluster readiness
            )
        except requests.RequestException as e:
            last_error = str(e)
            continue
        if resp.status_code < 500:
            break
        last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
    if resp is None:
        raise RuntimeError(f"ray-provisioner failed: {last_error}")
    if resp.status_code != 200:
        raise RuntimeError(
            f"ray-provisioner returned {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()["address"]


def inventory(token: str | None = None) -> list[dict]:
    """List available GPU models (model, count, memory, nodes)."""
    resp = requests.get(
        f"{config.PROVISIONER_URL}/inventory", headers=_headers(token), timeout=30
    )
    resp.raise_for_status()
    return resp.json()["models"]


def _print_inventory() -> None:
    models = inventory()
    print(f"{'MODEL':<24} {'CARDS':>5} {'MEM':>9}  NODES")
    for m in models:
        mem = f"{m['memory_mb'] / 1024:.1f} GiB"
        print(f"{m['model']:<24} {m['count']:>5} {mem:>9}  {', '.join(m['nodes'])}")
    print("\nUse as run_job(resources={'gpu_type': '<model>', ...})")


if __name__ == "__main__":
    _print_inventory()
