FROM node:22-slim@sha256:d415caac2f1f77b98caaf9415c5f807e14bc8d7bdea62561ea2fef4fbd08a73c

ARG CODEX_VERSION=0.162.1
ARG CLAUDE_VERSION=2.1.296
ARG AGY_VERSION=1.3.1
ARG AGY_URL=https://storage.googleapis.com/antigravity-public/antigravity-cli/1.3.1-4582356770750464/linux-arm/cli_linux_arm64.tar.gz
ARG AGY_SHA512=c41b8cd8c526eb043fa0b019377ab8109190b547624f17f5255868ace79cd8e9195a60ac7b89478101127effbe049341c4108ed0ead4b160968704a8b79d8799

RUN apt-get update \
    && apt-get install --no-install-recommends -y bubblewrap ca-certificates curl git python3 ripgrep dbus gnome-keyring libsecret-tools \
    && npm install --global @openai/codex@${CODEX_VERSION} @anthropic-ai/claude-code@${CLAUDE_VERSION} \
    && curl -fsSL "$AGY_URL" -o /tmp/agy.tar.gz \
    && printf '%s  %s\n' "$AGY_SHA512" /tmp/agy.tar.gz | sha512sum --check --status \
    && tar -xzf /tmp/agy.tar.gz -C /tmp antigravity \
    && install -o root -g root -m 0755 /tmp/antigravity /usr/local/bin/agy \
    && test "$(codex --version)" = "codex-cli ${CODEX_VERSION}" \
    && claude --version | head -1 | rg "^${CLAUDE_VERSION} " \
    && test "$(agy --version)" = "$AGY_VERSION" \
    && rm -rf /var/lib/apt/lists/* /root/.npm /tmp/agy.tar.gz /tmp/antigravity \
    && install -d -o node -g node -m 0700 /work /record /auth /auth/keyrings /home/node/.codex /run/user/1000

COPY --chmod=0555 docker/agents/worker.py docker/agents/agy_credentials.py src/agent/safe_artifacts.py src/agent/cli_events.py /opt/decompbench/
ENV XDG_RUNTIME_DIR=/run/user/1000
USER 1000:1000
WORKDIR /work
ENTRYPOINT ["python3", "-I", "-B", "/opt/decompbench/worker.py"]
