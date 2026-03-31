# myCode Docker image — full isolation for stress testing untrusted code.
#
# This Dockerfile is used by `mycode --containerised` to build the
# container image automatically.  You can also build it manually:
#
#   docker build -t mycode:py3.11 .
#   docker run --rm -v /path/to/project:/workspace/project:ro mycode:py3.11 /workspace/project --non-interactive --offline
#
# The --containerised flag handles all of this for you.

FROM python:3.11-slim

# Layer 1: System build tools + libraries for compiling Python C extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential libffi-dev libssl-dev python3-dev \
       libpq-dev libxml2-dev libxslt1-dev \
       libjpeg-dev zlib1g-dev libcurl4-openssl-dev \
       cargo \
       libopenblas-dev libgfortran5 git \
       ca-certificates curl gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Node.js 20 LTS via NodeSource (separate layer to avoid cache hits
# from the old Dockerfile that installed Debian's ancient nodejs/npm)
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
       > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version && npm --version

# Install myCode from local source
COPY . /opt/mycode
RUN pip install --no-cache-dir "/opt/mycode[web,pdf]"

# Create workspace directory
RUN mkdir -p /workspace

# MYCODE_UVICORN_WORKERS: defaults to 1. The in-memory job store is
# per-process, so workers > 1 requires a shared backend (Redis).
# PORT: Railway sets this dynamically; defaults to 8000 for local use.
CMD ["sh", "-c", "uvicorn mycode.web.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${MYCODE_UVICORN_WORKERS:-1}"]
