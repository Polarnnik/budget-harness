# python:3.12-slim, pinned by digest
FROM python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# The container runs with a read-only root: no bytecode or caches are written at run time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# requirements.txt is the full lock (uv pip compile requirements.in); every version is pinned.
# pip from the base image installs it, so the build needs only Docker Hub and PyPI.
COPY requirements.txt .
RUN pip install --no-deps --compile -r requirements.txt

COPY run_agent.py .
COPY harness ./harness
RUN python -m compileall -q /app

ENTRYPOINT ["python", "/app/run_agent.py"]
CMD ["--tasks-dir", "/data/tasks"]
