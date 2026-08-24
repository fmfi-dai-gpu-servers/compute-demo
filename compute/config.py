"""Platform constants for the FMFI DAI compute Ray deployment.

Single source of truth for service URLs, the Keycloak client used by the CLI,
and the environment-variable prefixes that should be forwarded to workers by
default. Change these here when the deployment moves.
"""

RAY_ADDRESS = "https://ray.c.dai.fmph.uniba.sk"

KEYCLOAK_URL = "https://auth.c.dai.fmph.uniba.sk"
REALM = "compute"
CLIENT_ID = "ray-cli"

DEVICE_URL = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/auth/device"
TOKEN_URL = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token"

# Credentials for these services live in the user's .env and are forwarded to
# worker pods via runtime_env["env_vars"] so jobs can reach S3 and MLflow.
PLATFORM_ENV_PREFIXES = ("S3_", "MLFLOW_")


PROVISIONER_URL = "https://provision.c.dai.fmph.uniba.sk"
# Treat a token as expired slightly before its real expiry to avoid races.
TOKEN_SKEW_SECONDS = 60
