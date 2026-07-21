FROM python:3.11-slim

ARG PYTORCH_VERSION=2.13.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/embedding/.cache/huggingface \
    LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    LOCAL_EMBEDDING_BATCH_SIZE=32 \
    LOCAL_EMBEDDING_DEVICE=cpu \
    EMBEDDING_SERVICE_HOST=0.0.0.0 \
    EMBEDDING_SERVICE_PORT=8090

WORKDIR /service

COPY requirements.txt ./requirements.txt

# Installing PyTorch from its CPU-only index prevents CUDA runtime packages
# from being included in the service image.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${PYTORCH_VERSION}" \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && addgroup --system embedding \
    && adduser --system --ingroup embedding --home /home/embedding embedding \
    && mkdir -p "${HF_HOME}" \
    && chown -R embedding:embedding /home/embedding

COPY --chown=embedding:embedding app.py ./app.py

USER embedding

EXPOSE 8090

HEALTHCHECK --interval=10s --timeout=5s --start-period=5m --retries=10 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('EMBEDDING_SERVICE_PORT', '8090') + '/health', timeout=5)" || exit 1

CMD ["sh", "-c", "exec uvicorn app:app --host \"${EMBEDDING_SERVICE_HOST}\" --port \"${EMBEDDING_SERVICE_PORT}\""]
