# app/Dockerfile

FROM python:3.9-slim

WORKDIR /tppsolver

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/somethington/tpp_solver/tree/main .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "tpp_solver_mt.py", "--server.port=8501", "--server.address=0.0.0.0"]