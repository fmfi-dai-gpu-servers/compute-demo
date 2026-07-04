#!/usr/bin/env python3
"""Ray CLI authentication via Keycloak device authorization grant.

Thin wrapper around :mod:`compute.auth` kept for the shell workflow:

    eval $(uv run python ray_auth.py)

This sets ``RAY_ADDRESS`` and ``RAY_AUTH_TOKEN`` in your shell. The actual
implementation lives in ``compute/auth.py``.
"""

from compute.auth import main

if __name__ == "__main__":
    main()
