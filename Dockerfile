# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /tppsolver

# Only the packages needed to build wheels and run the health check.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so the layer is cached across source changes.
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY . /tppsolver

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /tppsolver
USER appuser

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "tpp_solver_mt.py", "--server.port=8501", "--server.address=0.0.0.0"]
