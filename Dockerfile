## base image

FROM python:3.11-bookworm

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*


## working directory
WORKDIR /app

## instal python dependencies
# Copy requirements first (Docker layer caching — only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

## copy source code

COPY api/ ./api/
COPY generation/ ./generation/
COPY ingestion/ ./ingestion/
COPY retrieval/ ./retrieval/
COPY pipeline.py .
COPY fastapiMain.py .


## export port
EXPOSE 8000

## start FastAPI

CMD ["uvicorn", "fastapiMain:app", "--host", "0.0.0.0", "--port", "8000"]
