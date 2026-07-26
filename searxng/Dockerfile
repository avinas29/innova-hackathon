# syntax=docker/dockerfile:1.7
#
# SearXNG packaged for Render (or any PaaS that injects $PORT).
#
# The upstream image is already correct; this only does two things it cannot
# do for us:
#
#   1. Bakes in a settings.yml with `formats: [json]`. SearXNG serves HTML only
#      by default and answers API requests with HTTP 403 — the single most
#      common reason a self-hosted instance "works in a browser but not from
#      code".
#   2. Binds whatever port the platform hands us. The image listens on 8080;
#      Render assigns $PORT (typically 10000) and health-checks that port only.

FROM searxng/searxng:latest

USER root

# Config lives here by convention; the entrypoint reads it on start.
COPY settings.yml /etc/searxng/settings.yml
RUN chown -R searxng:searxng /etc/searxng 2>/dev/null || true

ENV SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml \
    INSTANCE_NAME=veritas-search \
    PORT=8080

# The upstream entrypoint honours SEARXNG_BIND_ADDRESS / SEARXNG_PORT, so the
# platform's $PORT is translated at start rather than baked into the image.
COPY --chmod=755 entrypoint.sh /usr/local/bin/veritas-searxng-entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/veritas-searxng-entrypoint.sh"]
