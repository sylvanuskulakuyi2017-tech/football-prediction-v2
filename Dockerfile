FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir --only-binary=:all: -r requirements.lock.txt \
    && python -m pip check
COPY . .
RUN python scripts/check_release.py
EXPOSE 8501
CMD ["python", "scripts/start_server.py"]
