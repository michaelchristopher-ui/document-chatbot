# The app alone. Inference is a separate process reached over HTTP — on this
# host or another — so nothing here has a model, a GPU or a driver in it, and
# the image stays the same one whichever topology it is deployed into. See
# docs/DEPLOY.md.
FROM python:3.11-slim

# libgomp is PyMuPDF's; the rest of the tree is pure wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ahead of the source so a code change does not re-resolve the dependency tree.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Everything the app writes lives here, and nothing it writes lives in the
# image: the volume outlives the container, which is the whole point of
# pointing VECTOR_URI, CATALOG_URI and ANALYTICS_URI at it below.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/documents \
    && chown -R app:app /data /app
USER app

ENV VECTOR_URI=/data/chatbot.db \
    CATALOG_URI=/data/catalog.db \
    ANALYTICS_URI=/data/analytics.db \
    DOCUMENTS_DIR=/data/documents \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Streamlit's own endpoint, and it answers as soon as the server is up — which
# is before the first ingest finishes. It says the web server is serving, not
# that the app can answer a question; what checks *that* is the embedding probe
# in `LLMProvider.embedding_dimension`, on the way to the chat window.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
