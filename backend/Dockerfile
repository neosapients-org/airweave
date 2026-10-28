# The taxonomy classifier is not part of this repository. ops/build-and-push.sh fetches it and
# passes it as the `classifier` build context; a build without one uses this empty stage and
# produces an image with no classifier (TAXONOMY_ENABLED=true then refuses to start).
FROM scratch AS classifier

FROM python:3.13.12-slim@sha256:8bc60ca09afaa8ea0d6d1220bde073bacfedd66a4bf8129cbdc8ef0e16c8a952

WORKDIR /app

# Install system dependencies and poetry in one optimized layer
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    unixodbc \
    unixodbc-dev \
    exiftool \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && pip install --no-cache-dir "poetry==2.3.2"

# Configure poetry for Docker environment
RUN poetry config virtualenvs.create false \
    && poetry config virtualenvs.in-project false \
    && poetry config cache-dir /opt/poetry-cache

# Copy poetry files first for better layer caching
COPY pyproject.toml poetry.lock ./

# Install dependencies with cache mount
RUN --mount=type=cache,target=/opt/poetry-cache \
    poetry install --only=main --no-interaction --no-ansi --no-root

# The taxonomy classifier (the discover and propose jobs run this image too). Its dependencies, including the
# build extra's, are locked in poetry.lock; the package goes in with --no-deps, and the requirements check fails the
# build if the lock and the package's ranges disagree.
# Mounted, not copied, so its source never becomes a layer; skipped when no classifier was given.
RUN --mount=type=bind,from=classifier,target=/tmp/neo-taxonomy-classifier,rw \
    --mount=type=bind,source=scripts/check_classifier_requirements.py,target=/tmp/check_classifier_requirements.py \
    if [ -f /tmp/neo-taxonomy-classifier/pyproject.toml ]; then \
      pip install --no-cache-dir --no-deps /tmp/neo-taxonomy-classifier \
      && python /tmp/check_classifier_requirements.py; \
    fi

# Copy entrypoint script with proper permissions
COPY --chmod=755 entrypoint.sh /app/entrypoint.sh

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
