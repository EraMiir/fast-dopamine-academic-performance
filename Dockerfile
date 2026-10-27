FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.4.2 \
    pyyaml==6.0.1 \
    psutil==5.9.8

COPY configs/ ./configs/
COPY data/ ./data/
COPY src/ ./src/

CMD ["python", "src/benchmark.py", "--config", "configs/config.yaml"]
