# app/Dockerfile

FROM python:3.10-slim

WORKDIR /tppsolver

RUN git clone https://github.com/somethington/tpp_solver .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "tpp_solver_mt.py", "--server.port=8501", "--server.address=0.0.0.0"]
