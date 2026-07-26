#!/bin/sh
# Translate the platform's $PORT into the variables the image actually reads,
# then hand off to the upstream entrypoint.
#
# The image serves via granian, which is configured by GRANIAN_* — not the
# SEARXNG_* names the docs imply. Verified from the image itself:
#     GRANIAN_PORT=8080
#     GRANIAN_HOST=::
#
# Render assigns a port at runtime and health-checks only that port. Without
# this the container stays on 8080, the probe finds nothing, and the deploy
# fails with no useful error.
set -eu

PORT="${PORT:-8080}"
export GRANIAN_PORT="$PORT"
# "::" is the IPv6 wildcard and accepts IPv4 too, which is what the image ships.
export GRANIAN_HOST="${GRANIAN_HOST:-::}"

# SearXNG will not start without a secret. Render supplies one via
# generateValue; this fallback keeps a plain `docker run` working locally.
if [ -z "${SEARXNG_SECRET:-}" ]; then
    SEARXNG_SECRET="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    export SEARXNG_SECRET
fi

echo "veritas-searxng: granian binding ${GRANIAN_HOST}:${GRANIAN_PORT}"

exec /usr/local/searxng/entrypoint.sh "$@"
