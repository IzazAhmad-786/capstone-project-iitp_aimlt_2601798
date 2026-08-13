import json
import os
from pathlib import Path
from typing import List

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

load_dotenv()

MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"
GROQ_MODEL = "llama-3.3-70b-versatile"
DOCS_DIR = Path(__file__).resolve().parent / "docs"
TOP_K = 3
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]

app = FastAPI(title="Zepto Support Assistant", version="1.0")

RAG_PROMPT_TEMPLATE = """### ROLE
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
{{"answer": "<answer as a string>", "sources": ["<doc id>", ...], "confidence": <float between 0 and 1>}}
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
{{"answer": "Standard delivery is free on orders above INR 149; orders below that
incur a flat INR 25 delivery fee.", "sources": ["doc_01"], "confidence": 0.95}}

Now answer the real customer question below using only the context above.
Customer question: {question}"""

DIRECT_PROMPT_TEMPLATE = """### ROLE
You are Zepto's customer support assistant.

### CONTEXT
The customer's question was classified as unrelated to Zepto's delivery, returns,
membership, tracking, cancellation, gift card, or support-hours policies, so no
policy documents were retrieved for it.

### TASK
Politely tell the customer you can only help with questions about Zepto policies.

### FORMAT
Respond with a single JSON object and nothing else:
{{"answer": "<short reply>", "sources": [], "confidence": <float between 0 and 1>}}

### LENGTH
One short sentence.

### NEGATIVE CONSTRAINT
Do not attempt to answer the general question itself, and do not invent a policy
that was never retrieved.

### FEW-SHOT EXAMPLE
Customer question: "What is the capital of France?"
Expected response:
{{"answer": "I can only answer questions about Zepto policies right now.", "sources": [], "confidence": 1.0}}

Now respond to the real customer question below.
Customer question: {question}"""

#  Task 1: load -> chunk -> embed -> store in ChromaDB
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"},
)

def load_and_index_data():
    ids, documents, metadatas = [], [], []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        ids.append(path.stem)
        documents.append(text)
        metadatas.append({"source": path.name})

    embeddings = embedding_model.encode(documents).tolist()
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {collection.count()} chunks into the '{collection.name}' ChromaDB collection")


load_and_index_data()


# 3. Optional MOCK_LLM=0 extension: Groq client (OpenAI-compatible) + validate-and-retry structured-output logic.
groq_client = None
if not MOCK_LLM:
    from openai import OpenAI

    groq_client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

def validate_structured_response(raw_response: str):
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        return None
    required_keys = ["answer", "sources", "confidence"]
    if not all(k in data for k in required_keys):
        return None
    if not isinstance(data["answer"], str):
        return None
    if not isinstance(data["sources"], list):
        return None
    if not isinstance(data["confidence"], (int, float)) or not (0 <= data["confidence"] <= 1):
        return None
    return data

def call_llm_with_retry(prompt: str, max_retries: int = 2) -> dict:
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries + 1):
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        data = validate_structured_response(raw)
        if data is not None:
            return data
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "That response was not valid JSON matching the required schema "
                '(keys: answer [string], sources [list], confidence [float 0-1]). '
                "Reply again with ONLY the corrected JSON object, no other text."
            ),
        })
    return {
        "answer": f"Error: could not obtain a valid structured response after {max_retries + 1} attempts.",
        "sources": [],
        "confidence": 0.0,
    }

def classify_intent_llm(query: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the customer question as exactly one word: "
                    "policy_question if it concerns Zepto's delivery, returns, "
                    "membership, order tracking, cancellation, gift cards, or "
                    "support hours policies, otherwise general_question. Reply "
                    "with only that one word."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    label = (response.choices[0].message.content or "").strip().lower()
    return "policy_question" if "policy_question" in label else "general_question"

# Task 3: State is a plain dict subclass, exactly the pattern used throughout the "Orchestration and Agent Workflow Design".
class SupportState(dict):
    query: str
    intent: str
    retrieved_ids: list
    retrieved_docs: list
    answer: str
    sources: list
    confidence: float

def classify_intent(state):
    query = state["query"]
    if MOCK_LLM:
        # Required, graded baseline: keyword heuristic, no LLM call.
        lowered = query.lower()
        state["intent"] = (
            "policy_question" if any(k in lowered for k in POLICY_KEYWORDS) else "general_question"
        )
    else:
        # Optional MOCK_LLM=0 extension: let the LLM classify instead.
        state["intent"] = classify_intent_llm(query)
    return state

def retrieve_and_answer(state):
    query = state["query"]
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    docs = results["documents"][0]
    ids = results["ids"][0]
    state["retrieved_docs"] = docs
    state["retrieved_ids"] = ids

    if MOCK_LLM:
        # Required, graded baseline: canned template built from the top chunk.
        top_chunk_snippet = docs[0][:200]
        state["answer"] = f"Based on the retrieved context: {top_chunk_snippet}"
        state["sources"] = list(ids)
        state["confidence"] = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt the real LLM, grounded in the
        # retrieved chunks, using the structured template above.
        context_block = "\n".join(f'[{cid}] "{doc}"' for cid, doc in zip(ids, docs))
        prompt = RAG_PROMPT_TEMPLATE.format(context=context_block, question=query)
        result = call_llm_with_retry(prompt)
        state["answer"] = result["answer"]
        state["sources"] = result["sources"]
        state["confidence"] = result["confidence"]
    return state

def direct_answer(state):
    if MOCK_LLM:
        # Required, graded baseline: fixed canned string, no LLM call.
        state["answer"] = "I can only answer questions about Zepto policies right now."
        state["sources"] = []
        state["confidence"] = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        prompt = DIRECT_PROMPT_TEMPLATE.format(question=state["query"])
        result = call_llm_with_retry(prompt)
        state["answer"] = result["answer"]
        state["sources"] = result["sources"]
        state["confidence"] = result["confidence"]
    return state

def route_after_classify(state):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

graph = StateGraph(SupportState)
graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve_and_answer", retrieve_and_answer)
graph.add_node("direct_answer", direct_answer)
graph.set_entry_point("classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    route_after_classify,
    {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
)
graph.add_edge("retrieve_and_answer", END)
graph.add_edge("direct_answer", END)
support_graph = graph.compile()


# Pydantic request/response models — Task 5's structured JSON schema
class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

# Task 6 -- FastAPI endpoints  
@app.get("/")
def home():
    return {"status": "running", "message": "Zepto Support Assistant Live", "mock_llm": MOCK_LLM}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    print(f"User question: {request.query}")
    result = support_graph.invoke({"query": request.query})
    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result["confidence"],
    )
