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

# Install Node.js for JavaScript project support
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libffi-dev libssl-dev python3-dev libopenblas-dev libgfortran5 git nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install myCode from local source
COPY . /opt/mycode
RUN pip install --no-cache-dir "/opt/mycode[web]"

# Create workspace directory
RUN mkdir -p /workspace

CMD ["uvicorn", "mycode.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
