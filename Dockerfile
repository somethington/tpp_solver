# syntax=docker/dockerfile:1

# ---- Builder: install dependencies as wheels (no compiler / apt needed) ----
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# All runtime deps ship manylinux wheels for cp311 (amd64 + arm64), so no
# build toolchain is required. Install into a relocatable target directory.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --target=/install -r requirements.txt

# ---- Runtime: distroless, non-root, no shell / package manager ----
FROM gcr.io/distroless/python3-debian12:nonroot

WORKDIR /app

# Third-party packages (from the builder) and the application source.
COPY --from=builder /install /app/site-packages
COPY . /app

# distroless python is 3.11; match the builder. Writable caches go to /tmp
# because the image runs as the unprivileged "nonroot" user.
ENV PYTHONPATH=/app/site-packages:/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER nonroot
EXPOSE 8501

# No curl in distroless: probe the Streamlit health endpoint with urllib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python3", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"]

ENTRYPOINT ["python3", "-m", "streamlit", "run", "tpp_solver_mt.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
