FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Pull CPU-only torch first. sentence-transformers would otherwise drag in the
# CUDA build, which is several GB we never use on this stack.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY kb ./kb

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /models \
    && chown -R appuser:appuser /models /srv
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
