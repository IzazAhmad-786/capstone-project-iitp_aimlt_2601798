# Module 3 - Zepto Support Assistant

A small RAG service for Zepto customer support. It answers policy questions (delivery, returns, membership, etc.) using Zepto's own docs, and politely declines anything else. Built with LangGraph + ChromaDB + FastAPI.

Runs fully offline by default — no API key, no signup, no network calls. That's the mode this gets graded on.

## Stack

- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`), runs locally
- **Vector store:** ChromaDB (in-memory)
- **Orchestration:** LangGraph `StateGraph`
- **API:** FastAPI + Pydantic
- **Optional real LLM:** Groq (`llama-3.3-70b-versatile`), only if `MOCK_LLM=0`

## Setup

```bash
cd support_assistant

python -m venv .venv
source .venv\Scripts\Activate.ps1
pip install -r requirements.txt

uvicorn main:app --reload
```

First run downloads the embedding model (~90MB), then it's cached. App runs on `http://localhost:8000`.

## MOCK_LLM toggle

One env var controls everything:

- **Unset / `MOCK_LLM=1` (default, graded):** no LLM calls anywhere. Intent is classified by keyword matching, and answers are canned templates built from retrieved chunks.
- **`MOCK_LLM=0` (optional extension):** calls Groq for classification and answer generation. Needs `GROQ_API_KEY` in `.env`.

## Architecture

**Ingestion + embedding** — `load_and_index_data()` reads all 8 files in `docs/`, one file = one chunk (they're short enough). Each chunk gets embedded with `all-MiniLM-L6-v2` and stored in the ChromaDB collection `zepto_policies`, with the doc filename as metadata.

**Retrieval** — happens inside the `retrieve_and_answer` node. Query gets embedded the same way, then `collection.query()` pulls the top-3 closest chunks. This always runs for real, mock mode or not — no LLM involved in retrieval itself.

**Generation** — this is the only part that checks `MOCK_LLM`:
- `classify_intent`: keyword match (mock) vs LLM call (real)
- `retrieve_and_answer`: canned `"Based on the retrieved context: ..."` string (mock) vs LLM answer grounded in the retrieved chunks (real)
- `direct_answer`: fixed string, no retrieval (mock) vs direct LLM call (real)

**Flow:** `POST /ask` → `classify_intent` → conditional edge → `retrieve_and_answer` (policy question) or `direct_answer` (general question) → response validated against the `AskResponse` schema → JSON back to client.

## LangGraph nodes

3 nodes, 1 conditional edge:

```
classify_intent → (policy_question?) → retrieve_and_answer → END
                → (general_question?) → direct_answer       → END
```

State is a plain `dict` (`SupportState`) carrying `query`, `intent`, `retrieved_ids`, `retrieved_docs`, `answer`, `sources`, `confidence`.

## Prompt template

Used only when `MOCK_LLM=0`. Follows role / context / task / format / length, plus a negative constraint and a few-shot example:

```
### ROLE
You are Zepto's customer support policy assistant. You answer customer questions
strictly from Zepto's official policy documents. You never speak as a
general-purpose assistant and you never invent policy details.

### CONTEXT
Zepto is a quick-commerce grocery and household-essentials delivery service. Below
is the retrieved context for this question — the only source of truth you are
allowed to use. Each chunk is labelled with the id of the source document it came
from.

{context}

### TASK
Read the customer's question and the retrieved context above. Answer the question
using only facts stated in that context. If the context does not contain enough
information to answer, say so explicitly rather than guessing.

### FORMAT
Respond with a single JSON object and nothing else, matching exactly this shape:
{"answer": "<answer as a string>", "sources": ["<doc id>", ...], "confidence": <float between 0 and 1>}
Do not include markdown, code fences, explanations, or any text outside the JSON object.

### LENGTH
Keep "answer" to 1-3 sentences.

### NEGATIVE CONSTRAINT
Do not answer using information not present in the provided context. Do not rely on
prior/general knowledge about Zepto, other delivery companies, or anything not
stated above. If the context does not cover the question, set "answer" to a short
statement that the information is not available in Zepto's policies, set "sources"
to an empty list, and set "confidence" to a low value such as 0.2.

### FEW-SHOT EXAMPLE
Customer question: "How much does standard delivery cost on a small order?"
Retrieved context:
[doc_01] "Zepto delivers grocery and household essentials to serviceable pin codes
within 10 to 30 minutes of order confirmation ... Standard delivery is free on
orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. ..."
Expected response:
{"answer": "Standard delivery is free on orders above INR 149; orders below that
incur a flat INR 25 delivery fee.", "sources": ["doc_01"], "confidence": 0.95}

Now answer the real customer question below using only the context above.
Customer question: {question}
```

A second, shorter version (`DIRECT_PROMPT_TEMPLATE`) handles the no-retrieval case the same way.

## Structured output

`AskResponse` (Pydantic): `answer: str`, `sources: list[str]`, `confidence: float` (0–1). Mock mode fills this in directly from code. Real-LLM mode validates the raw JSON (`validate_structured_response`) and retries up to 2 times with corrective feedback (`call_llm_with_retry`) before returning a marked error.

## Example calls

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "How much does delivery cost?"}'
```
```json
{"answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation...", "sources": ["doc_01", "doc_04", "doc_05"], "confidence": 1.0}
```

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```
```json
{"answer": "I can only answer questions about Zepto policies right now.", "sources": [], "confidence": 1.0}
```

*(Swap these for your own terminal output before submitting — exact chunk text may vary slightly.)*

## Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

Serves the same `/ask` endpoint on `http://localhost:7860`, `MOCK_LLM=1` by default.

## Optional: real LLM mode

```bash
# .env
MOCK_LLM=0
GROQ_API_KEY=your_key_here
```

Free key from [console.groq.com](https://console.groq.com/keys). Not required for grading — mock mode is the baseline.
