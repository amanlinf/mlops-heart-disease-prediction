# uses base image for python 3.10
FROM python:3.10-slim

# setting up env variables
# model path and metadata paths are where the model and metadata sits insides the docker container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model.joblib \
    METADATA_PATH=/app/artifacts/metadata.json

# docker container working dir
WORKDIR /app
# copying requirements-api.txt into current working dir, have separated the requirements for building the fastapi image
COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir -r requirements-api.txt \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser

# giving permissions for the non-root user
COPY --chown=10001:10001 api/ ./api/
COPY --chown=10001:10001 src/ ./src/
# need to run src.train before docker build
COPY --chown=10001:10001 artifacts/model.joblib artifacts/metadata.json ./artifacts/

# running container application as a non root user
USER 10001
EXPOSE 8000
# health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200" || exit 1
# running the fastapi server using asgi uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
