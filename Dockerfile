# Reproducible environment for volbench.
# Build:  docker build -t volbench .
# Test:   docker run --rm volbench
# Repro:  docker run --rm -v "$PWD/results:/app/results" volbench make reproduce
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: make for the reproduce target; build tools for any wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends make build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pinned dependencies first (better layer caching), matching CI exactly.
COPY requirements.lock pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -r requirements.lock && pip install -e . --no-deps

# Copy the rest (data, scripts, tests, report).
COPY . .

# Default: run the test suite.
CMD ["pytest", "-q"]
