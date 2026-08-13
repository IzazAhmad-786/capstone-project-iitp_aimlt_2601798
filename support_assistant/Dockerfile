FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first so this layer is cached across rebuilds
# that only touch application code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the all-MiniLM-L6-v2 embedding model at build time so the
# container never needs network access at runtime to serve requests.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Application code and the policy corpus it embeds/indexes.
COPY docs/ ./docs/
COPY main.py .

# Graded baseline: MOCK_LLM defaults to mock mode, no key required.
ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
