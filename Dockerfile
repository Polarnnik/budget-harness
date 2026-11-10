# uv: fast installer with parallel downloads, pinned by digest.
FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

# python:3.12-slim, pinned by digest
FROM python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# The container runs with a read-only root: no bytecode or caches are written at run time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app

# requirements.txt is the full lock (uv pip compile requirements.in); every version is pinned.
COPY requirements.txt .
RUN uv pip install --system --no-deps --compile-bytecode -r requirements.txt

COPY run_agent.py .
COPY harness ./harness
RUN python -m compileall -q /app

ENTRYPOINT ["python", "/app/run_agent.py"]
CMD ["--tasks-dir", "/data/tasks"]
