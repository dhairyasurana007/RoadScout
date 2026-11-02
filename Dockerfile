# RoadScout hosted demo (read-only). Built by Render from the repo; see "Hosted demo (Render)" in README.md.
# Stage 1 builds the frontend, stage 2 collects the optional prebuilt dataset, stage 3 is the server.

# ---- 1. frontend build -------------------------------------------------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 2. the demo dataset (demo/demo.sqlite and demo/storage/, added later); the build works without demo/ -------------
# "demo*" matches nothing when the folder is absent, and a COPY with one source that exists (requirements.txt) and a
# wildcard that matches nothing still succeeds. requirements.txt is removed again, leaving only the contents of demo/.
FROM python:3.12-slim AS demo-data
WORKDIR /stage
COPY requirements.txt demo* ./
RUN rm requirements.txt && mkdir -p /out && cp -r /stage/. /out/

# ---- 3. the server -----------------------------------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ROADSCOUT_DEMO=1 \
    ROADSCOUT_STATIC_DIR=/app/frontend/dist \
    ROADSCOUT_DB=/app/demo/demo.sqlite \
    ROADSCOUT_STORAGE_DIR=/app/demo/storage
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Only what the server needs: no tests, no footage, no .env, no labels (see .dockerignore).
COPY backend/ backend/
COPY config/ config/
COPY --from=frontend /build/dist frontend/dist
RUN mkdir -p demo/storage
COPY --from=demo-data /out/ demo/

RUN useradd --create-home --uid 10001 roadscout && chown -R roadscout /app/demo
USER roadscout

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'))"
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
