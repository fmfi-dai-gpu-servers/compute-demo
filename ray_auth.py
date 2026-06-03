#!/usr/bin/env python3
"""Ray CLI authentication via Keycloak device authorization grant.

Gets an access token from the compute realm using the ray-cli public client.
The token can be used with Ray's JobSubmissionClient to submit jobs to the
authenticated Ray cluster.

Usage:
    # Get a token (prints to stdout)
    python ray_auth.py

    # Use in Python
    from ray.job_submission import JobSubmissionClient
    token = ...  # from ray_auth.py
    client = JobSubmissionClient(
        address="https://ray.c.dai.fmph.uniba.sk",
        headers={"Authorization": f"Bearer {token}"}
    )
    client.submit_job(entrypoint="python my_job.py")
"""

import base64
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.parse
import urllib.request

KEYCLOAK_URL = "https://auth.c.dai.fmph.uniba.sk"
REALM = "compute"
CLIENT_ID = "ray-cli"
RAY_ADDRESS = "https://ray.c.dai.fmph.uniba.sk"

DEVICE_URL = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/auth/device"
TOKEN_URL = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token"


def _generate_pkce():
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).decode("ascii").rstrip("=")
    return verifier, challenge


def get_token():
    code_verifier, code_challenge = _generate_pkce()

    data = urllib.parse.urlencode({
        "client_id": CLIENT_ID,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
        "scope": "openid profile email",
    }).encode()

    req = urllib.request.Request(DEVICE_URL, data=data)
    resp = urllib.request.urlopen(req)
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

        poll_data = urllib.parse.urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_id": CLIENT_ID,
            "device_code": device_code,
            "code_verifier": code_verifier,
        }).encode()

        req = urllib.request.Request(TOKEN_URL, data=poll_data)
        try:
            resp = urllib.request.urlopen(req)
            tokens = json.loads(resp.read())
            return tokens["access_token"]
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


def main():
    token = get_token()
    print(f"export RAY_ADDRESS={RAY_ADDRESS}")
    print(f"export RAY_AUTH_TOKEN={token}")


if __name__ == "__main__":
    main()
