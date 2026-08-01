FROM python:3.11-slim
RUN apt-get update && apt-get install -y gcc g++ git curl ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p checkpoints tokenizer/vocab models outputs/images outputs/audio memory logs api_keys
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "codemind.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
