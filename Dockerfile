FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for compiling python extensions (like sqlite/postgresql packages if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir psycopg2-binary gunicorn

COPY . .

# Run gunicorn server on port 8080 (Cloud Run default)
EXPOSE 8080
CMD ["gunicorn", "app.api.main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080"]
