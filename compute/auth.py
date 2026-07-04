"""Keycloak device-flow authentication for the Ray CLI.

Gets an access token from the compute realm using the ray-cli public client.
The token authenticates :class:`~ray.job_submission.JobSubmissionClient`
requests to the Ray cluster behind oauth2-proxy.

Typical use does not call these functions directly — :class:`compute.ComputeClient`
calls :func:`ensure_token` automatically. They are exposed for the
``eval $(python ray_auth.py)`` shell workflow and for callers that want to
manage tokens themselves.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config

ENV_TOKEN = "RAY_AUTH_TOKEN"
ENV_ADDRESS = "RAY_ADDRESS"


def _generate_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(32)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode("ascii")
        .rstrip("=")
    )
    return verifier, challenge


def get_token() -> str:
    """Run the OAuth2 device authorization grant and return the access token.

    Prints a verification URL + user code to stderr and blocks until the user
    authenticates in their browser (Keycloak → GitHub), then returns the
    Keycloak access token.
    """
    code_verifier, code_challenge = _generate_pkce()

    data = urllib.parse.urlencode(
        {
            "client_id": config.CLIENT_ID,
            "code_challenge_method": "S256",
            "code_challenge": code_challenge,
            "scope": "openid profile email",
        }
    ).encode()

    resp = urllib.request.urlopen(urllib.request.Request(config.DEVICE_URL, data=data))
    result = json.loads(resp.read())

    device_code = result["device_code"]
    user_code = result["user_code"]
    verification_uri = result.get(
        "verification_uri_complete", result["verification_uri"]
    )
    interval = result.get("interval", 5)
    expires_in = result.get("expires_in", 600)

    print(f"\n  Authenticate at: {verification_uri}", file=sys.stderr)
    print(f"  Code: {user_code}", file=sys.stderr)
    print(f"  (expires in {expires_in}s)\n", file=sys.stderr)

    deadline = time.time() + expires_in
    while time.time() < deadline:
        time.sleep(interval)

        poll_data = urllib.parse.urlencode(
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": config.CLIENT_ID,
                "device_code": device_code,
                "code_verifier": code_verifier,
            }
        ).encode()

        try:
            resp = urllib.request.urlopen(
                urllib.request.Request(config.TOKEN_URL, data=poll_data)
            )
            return json.loads(resp.read())["access_token"]
        except urllib.error.HTTPError as e:
            body = json.loads(e.read())
            error = body.get("error", "")
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5
                continue
            if error == "expired_token":
                print("Device code expired. Try again.", file=sys.stderr)
                sys.exit(1)
            if error == "access_denied":
                print("Access denied.", file=sys.stderr)
                sys.exit(1)
            print(f"Error: {body}", file=sys.stderr)
            sys.exit(1)

    print("Timed out waiting for authentication.", file=sys.stderr)
    sys.exit(1)


def is_token_expired(token: str) -> bool:
    """Return True if the JWT's ``exp`` claim has passed (with a small skew).

    Only decodes the token locally — no network call. A malformed token is
    treated as expired so the caller re-authenticates.
    """
    parts = token.split(".")
    if len(parts) < 2:
        return True
    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8")
        )
        exp = payload.get("exp")
        return exp is None or time.time() > exp - config.TOKEN_SKEW_SECONDS
    except Exception:
        return True


def ensure_token() -> str:
    """Return a usable access token, re-authenticating via device flow if needed.

    Reads ``RAY_AUTH_TOKEN`` from the environment; if it is missing or expired,
    runs :func:`get_token`, exports it back to the environment so subprocesses
    (and subsequent calls) reuse it, and returns it.
    """
    token = os.environ.get(ENV_TOKEN)
    if token and not is_token_expired(token):
        return token

    if token:
        print("Ray token expired — re-authenticating.", file=sys.stderr)
    token = get_token()
    os.environ[ENV_TOKEN] = token
    return token


def print_exports() -> None:
    """Print ``export`` statements for the shell, as the CLI entry point."""
    token = ensure_token()
    print(f"export {ENV_ADDRESS}={config.RAY_ADDRESS}")
    print(f"export {ENV_TOKEN}={token}")


def main() -> None:
    print_exports()


if __name__ == "__main__":
    main()
