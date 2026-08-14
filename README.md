# capstone-project-iitp_aimlt_2601798

End-to-end "Zepto Data & AI Platform" capstone for the IIT Patna AI/ML certificate program. One repo, three modules that build on top of each other — a scraping-to-SQL data pipeline, a customer analytics/ML pipeline, and a RAG-based GenAI support assistant. This file covers setup, installation, and usage for all three, start to finish.

---

## 1. What's in here

| # | Module | Folder | What it does |
|---|---|---|---|
| 1 | Data Pipeline | `/data_pipeline` | Scrapes book listings from books.toscrape.com, cleans and converts them, loads them into SQLite, and runs SQL + pandas queries on the result. |
| 2 | Analytics / ML | `/analytics` | EDA and classification/regression modeling on the Titanic dataset — profiling, cleaning, 3 classifiers, imbalance handling, tuning, and a saved end-to-end pipeline. |
| 3 | Support Assistant | `/support_assistant` | A RAG-based customer support assistant (LangGraph + ChromaDB + FastAPI) that answers policy questions from a small doc set, with an offline "mock LLM" mode and an optional real-LLM mode via Groq. |

The three modules map to the three technical parts of the curriculum (data foundations, classical ML, and generative AI/agents). They don't share code or a runtime — each has its own setup and can be run independently.

```
capstone-project-iitp_aimlt_2601798-main/
├── README.md
├── requirements.txt             (optional, installs all three modules at once)
├── data_pipeline/
│   ├── data_pipeline.py
│   ├── books.db                  (sample saved run)
│   └── output.txt                (sample saved run)
├── analytics/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── titanic.csv               (saved on first run)
│   ├── titanic_full_pipeline.joblib
│   └── charts/
└── support_assistant/
    ├── main.py
    ├── requirements.txt
    ├── Dockerfile
    ├── .env.example
    └── docs/                      (policy corpus, 8 files)
```

---

## 2. Prerequisites

- **Python 3.10 or 3.11** (the support assistant's Docker image is built on `python:3.11-slim`; anything 3.10+ works fine locally for all three modules)
- **pip** and (recommended) **venv** for isolated environments — the three modules use different, non-overlapping dependencies, so a separate virtual environment per module avoids version conflicts
- **Git**, to clone the repo
- **Docker** — optional, only needed to run the support assistant in a container instead of locally
- No API keys are required for the default/graded path of any module. The support assistant only needs a key (Groq, free tier) if you switch it out of mock mode.

Clone the repo first:

```bash
git clone <repo-url>
cd capstone-project-iitp_aimlt_2601798-main
```

Everything below assumes you're running commands from inside that folder unless a `cd` is shown.

---

## 3. Module 1 — Data Pipeline (`/data_pipeline`)

Scrapes book data from **books.toscrape.com**, cleans it, converts the price to INR, loads it into a normalized SQLite database, and runs SQL + pandas queries on it.

### 3.1 Data source

- Website: https://books.toscrape.com/ — a public site made only for scraping practice, no login or API key needed.
- Scrapes the first **5 categories** (change with `NUM_CATEGORIES` in the script).
- Per book: `title`, `price`, `star_rating`, `availability`, `category`.

### 3.2 Setup

```bash
cd data_pipeline

pip install requests beautifulsoup4 pandas
```

(Python's `sqlite3` module is built in, no install needed.)

### 3.3 Run

```bash
python data_pipeline.py
```

This single command:

1. Scrapes the book data from the website.
2. Cleans and converts the data.
3. Creates/rebuilds the SQLite database (`books.db`).
4. Loads the cleaned data into the database.
5. Runs 5 SQL queries and prints the results.
6. Re-runs 2 of those queries with `pd.read_sql`.
7. Reproduces the JOIN query using `pd.merge` (no SQL) and checks that both results match.

To save the output to a file (recommended for submission):

```bash
python data_pipeline.py > output.txt
```

### 3.4 Data cleaning decisions

| Field | Raw format | Cleaned format | How |
|---|---|---|---|
| `price` | `"£51.77"` | `price_gbp` (float) | Removed the `£` symbol and a stray encoding character (`Â`), then converted to a number. |
| `star_rating` | Text class like `"Three"` | `rating` (int 1–5) | Mapped word → number using a dictionary (`One`→1 … `Five`→5). |
| `availability` | `"In stock (22 available)"` | `in_stock` (True/False) | Checked whether the text contains `"in stock"`. |

**Missing/bad values:** if `price_gbp` or `rating` couldn't be parsed for a row, that value was **median-imputed** rather than dropping the row — losing a full book record is a bigger loss than approximating one number, and the dataset is small (needs ≥60 rows), so dropping the required minimum by accident wasn't worth risking.

### 3.5 Currency conversion

A fixed, project-defined rate: `1 GBP = 105.50 INR` — a constant for this assignment only, not a live or historical market rate, so no API call or date lookup is needed.

```
price_inr = price_gbp * 105.50
```

### 3.6 Database schema

SQLite database file: `books.db` (regenerated automatically each time the script runs).

**Table `categories`**

| Column | Type |
|---|---|
| category_id | INTEGER PRIMARY KEY |
| category_name | TEXT UNIQUE |

**Table `books`**

| Column | Type |
|---|---|
| book_id | INTEGER PRIMARY KEY |
| title | TEXT |
| price_gbp | REAL |
| price_inr | REAL |
| rating | INTEGER |
| in_stock | BOOLEAN |
| category_id | INTEGER (FOREIGN KEY → categories.category_id) |

Indexes on `category_name`, `category_id`, `price_inr`, `rating`, and `in_stock` for faster queries.

### 3.7 SQL queries

All 5 queries are in `run_queries()` inside `data_pipeline.py`, and are printed when the script runs.

| # | Purpose | SQL feature shown |
|---|---|---|
| Q1 | In-stock books over ₹300 | SELECT / WHERE |
| Q2 | 10 most expensive books | ORDER BY / LIMIT |
| Q3 | Distinct category names | DISTINCT |
| Q4 | 4–5 star books priced ₹500–₹2000 | IN / BETWEEN |
| Q5 | Top 10 highest-rated books per category | JOIN (with window function) |

### 3.8 Pandas verification

- `pd.read_sql` re-runs **Q1** and **Q2** directly into DataFrames.
- The **Q5 (JOIN)** result is reproduced **without SQL**, using `pd.merge()` on the in-memory `books` and `categories` DataFrames, plus a manual ranking step (`groupby` + `cumcount`) to copy what `ROW_NUMBER()` did in SQL.
- The script compares both results with `.equals()` and prints `SQL result matches pandas merge result: True`.

### 3.9 Files in this module

- `data_pipeline.py` — full pipeline (scrape → clean → convert → load → query)
- `books.db` — SQLite database (auto-created by the script)
- `output.txt` — saved output of all queries (created by running the script with `>`)

---

## 4. Module 2 — Analytics / ML (`/analytics`)

Titanic dataset, loaded once (`sns.load_dataset('titanic')`) and split across two notebooks. Everything downstream builds on the same cleaned data.

### 4.1 Setup

```bash
cd analytics

pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib jupyter nbformat
```

### 4.2 Run

```bash
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
```

Or open both notebooks in Jupyter/VS Code and run them top to bottom, **in order**. `01_eda.ipynb` must run first — it's the only place `sns.load_dataset` is called, and it saves the raw data to `titanic.csv` right after loading, so `02_modeling.ipynb` just reads that CSV instead of hitting the network again.

### 4.3 Files

| File | What it is |
|---|---|
| `01_eda.ipynb` | Load, profile, clean, EDA |
| `02_modeling.ipynb` | Split, preprocessing, 3 classifiers, imbalance handling, tuning, regression, saved model |
| `titanic.csv` | Raw dataset, saved once before any cleaning |
| `titanic_full_pipeline.joblib` | Fitted preprocessing + tuned Random Forest, bundled |
| `charts/` | PNGs referenced below |

(Write-ups live in this README rather than in notebook markdown — the notebooks are kept to code + output only.)

### 4.4 Part A — Profiling, cleaning, data story

**Load & profile.** Shape: `(891, 15)`. Saved raw to `titanic.csv` before touching anything.

Missing values:

| Column | Missing | % |
|---|---|---|
| `deck` | 688 | 77.22% |
| `age` | 177 | 19.87% |
| `embarked` | 2 | 0.22% |
| `embark_town` | 2 | 0.22% |

**Missing-value handling.** Rule: <5% → drop rows, 5–30% → impute, >30% → drop column.

- `embarked` (0.22%) — dropped 2 rows.
- `embark_town` (0.22%, same rows) — cleared automatically, it's just `embarked` spelled out.
- `age` (19.87%) — median-imputed (median holds up better than mean given the right-skew).
- `deck` (77.22%) — dropped the column; imputing 3/4 of it would just be making up data.

Result: 889 rows, 13 columns.

**`age`/`fare` outliers & skew** (`charts/01_univariate_age_fare.png`):

| Column | IQR outliers | Bounds |
|---|---|---|
| `age` | 65 | [2.50, 54.50] |
| `fare` | 114 | [-26.76, 65.66] |

`fare` mean/median/mode: **32.10 / 14.45 / 8.05** → mean > median > mode = right-skewed (long tail of expensive tickets).

**Bivariate + correlation.**

- Survival by sex: female 0.740, male 0.189
- Survival by class: 1st 0.626, 2nd 0.473, 3rd 0.242
- Survival by sex+class: female-1st 0.967, male-1st 0.369, female-2nd 0.921, male-2nd 0.157, female-3rd 0.500, male-3rd 0.135

`charts/02_correlation_heatmap.png` — 6 numeric columns only (`survived, pclass, age, sibsp, parch, fare`); `adult_male`/`alone` skipped since they're derived from columns already in the matrix.

Top 2 correlations: **`pclass`/`fare` (-0.55)** — fare basically sets class, expected. **`sibsp`/`parch` (+0.41)** — both are proxying for "traveling with family."

**Data story (4 charts).** Story: sex mattered most, class/fare second, age only mattered much for kids.

1. **Survival by class & sex** (`03_survival_by_class_sex.png`) — drops from women→men in every class and 1st→3rd in every sex. Women in 1st/2nd: 90%+. Men in 3rd: ~15%.
2. **Age by survival** (`04_age_by_survival.png`) — medians similar, but survivors show a cluster of young kids. Age alone is a weak signal.
3. **Fare vs age, by survival** (`05_fare_age_scatter.png`) — survivors cluster at higher fares across age groups. Fare mattered more than age.
4. **Pairplot** (`06_pairplot.png`) — `pclass`/`fare` visibly separate the two groups; `age` distributions mostly overlap.

**Standardization check.** EDA-only, doesn't feed the model (that's fit on the train split later).

| | age | fare |
|---|---|---|
| mean before | 29.32 | 32.10 |
| std before | 12.98 | 49.70 |
| mean after | 0.00 | 0.00 |
| std after | 1.00 | 1.00 |

Works as expected (`07_standardization_check.png`). `fare` keeps its skew shape — scaling doesn't fix skew.

### 4.5 Part B — Modeling

Reads raw `titanic.csv`, not the Part A cleaned version — Part A's median fill used the full dataset, which would leak into the test set if reused for modeling. So this notebook does its own imputation/encoding/scaling, fit on train only.

**Split.** Features: `pclass, sex, age, sibsp, parch, fare, embarked`. Target: `survived`. Class balance ~62/38, moderate imbalance — stratified on `survived` so both splits keep that ratio (checked: train 61.7/38.3, test 61.5/38.5). 80/20, `random_state=42`.

**Preprocessing.** `age` → median impute. `embarked` → mode impute. `sex`/`embarked` → one-hot (`drop='first'`, `handle_unknown='ignore'`). Numeric cols → `StandardScaler`. All inside a `ColumnTransformer` + `Pipeline`, so `.fit()` only touches `X_train`, test only gets `.transform()`.

**Train & evaluate 3 classifiers.** Logistic Regression, Decision Tree (`max_depth=4`), Random Forest (`n_estimators=200`) — same split. Tree plot: `08_decision_tree.png`. Confusion matrices: `09_confusion_matrices.png`. ROC: `10_roc_curves.png`.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8435 |
| Decision Tree | 0.7765 | 0.8085 | 0.5507 | 0.6552 | 0.8144 |
| Random Forest | 0.8101 | 0.7778 | 0.7101 | 0.7424 | 0.8310 |

Random Forest wins F1/accuracy, Logistic Regression wins AUC. Decision tree trails on most metrics — a single depth-4 tree can't capture interactions like sex×class.

**Imbalance handling.** Baseline vs `class_weight='balanced'` vs SMOTE (train-fold only, via `imblearn` pipeline), on Random Forest:

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.7778 | 0.7101 | 0.7424 |
| `class_weight='balanced'` | 0.7966 | 0.6812 | 0.7344 |
| SMOTE | 0.7286 | 0.7391 | 0.7338 |

None clearly beats baseline — they just trade precision for recall in opposite directions. Given the imbalance isn't severe, pick based on whether false negatives or false positives cost more.

**Tuning.** `GridSearchCV` on Random Forest (`n_estimators`, `max_depth`, `max_features`, `cv=5`). `oob_score=True` set at construction (required for `oob_score_` to populate).

Best params: `max_depth=4, max_features='sqrt', n_estimators=200`. Best CV accuracy: 0.8301. OOB: 0.8287. Test accuracy: 0.7933. All close together — not badly overfit.

**Regression (predicting `fare`).** Linear regression, own 80/20 split. Residuals: `11_residual_plot.png`.

| MAE | RMSE | R² | Adj R² |
|---|---|---|---|
| 20.90 | 30.53 | 0.3975 | 0.3729 |

Residuals fan out wider as predicted fare increases — heteroscedasticity, not constant variance. Matches `fare`'s right-skew from the univariate check.

**Final comparison + recommendation.**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8435 |
| Decision Tree | 0.7765 | 0.8085 | 0.5507 | 0.6552 | 0.8144 |
| Random Forest | 0.8101 | 0.7778 | 0.7101 | 0.7424 | 0.8310 |

| Model | MAE | RMSE | R² | Adj R² |
|---|---|---|---|---|
| Linear Regression (fare) | 20.90 | 30.53 | 0.3975 | 0.3729 |

**Recommendation: tuned Random Forest.** Best F1/accuracy, close enough on AUC (0.8310 vs 0.8435) that it's not a deciding factor. CV accuracy and OOB score both close to test accuracy — not overfit. Logistic Regression is the fallback if interpretability matters — performance is nearly identical.

**Saved pipeline.** The whole pipeline (preprocessing + tuned Random Forest) is saved as one object via `joblib.dump`, not just the bare model. Reloaded with `joblib.load`, it runs on raw untouched rows and predictions match the original — works end-to-end on new raw data, no manual preprocessing needed.

---

## 5. Module 3 — Support Assistant (`/support_assistant`)

A small RAG service for Zepto customer support. It answers policy questions (delivery, returns, membership, etc.) using Zepto's own docs, and politely declines anything else. Built with LangGraph + ChromaDB + FastAPI.

Runs fully offline by default — no API key, no signup, no network calls. That's the mode this gets graded on.

### 5.1 Stack

- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`), runs locally
- **Vector store:** ChromaDB (in-memory)
- **Orchestration:** LangGraph `StateGraph`
- **API:** FastAPI + Pydantic
- **Optional real LLM:** Groq (`llama-3.3-70b-versatile`), only if `MOCK_LLM=0`

### 5.2 Setup

```bash
cd support_assistant
python -m venv .venv
pip install -r requirements.txt

uvicorn main:app --reload
```

First run downloads the embedding model (~90MB), then it's cached. App runs on `http://localhost:8000`.

### 5.3 MOCK_LLM toggle

One env var controls everything:

- **Unset / `MOCK_LLM=1` (default, graded):** no LLM calls anywhere. Intent is classified by keyword matching, and answers are canned templates built from retrieved chunks.
- **`MOCK_LLM=0` (optional extension):** calls Groq for classification and answer generation. Needs `GROQ_API_KEY` in `.env`.

### 5.4 Architecture

**Ingestion + embedding** — `load_and_index_data()` reads all 8 files in `docs/`, one file = one chunk (they're short enough). Each chunk gets embedded with `all-MiniLM-L6-v2` and stored in the ChromaDB collection `zepto_policies`, with the doc filename as metadata.

**Retrieval** — happens inside the `retrieve_and_answer` node. Query gets embedded the same way, then `collection.query()` pulls the top-3 closest chunks. This always runs for real, mock mode or not — no LLM involved in retrieval itself.

**Generation** — this is the only part that checks `MOCK_LLM`:
- `classify_intent`: keyword match (mock) vs LLM call (real)
- `retrieve_and_answer`: canned `"Based on the retrieved context: ..."` string (mock) vs LLM answer grounded in the retrieved chunks (real)
- `direct_answer`: fixed string, no retrieval (mock) vs direct LLM call (real)

**Flow:** `POST /ask` → `classify_intent` → conditional edge → `retrieve_and_answer` (policy question) or `direct_answer` (general question) → response validated against the `AskResponse` schema → JSON back to client.

### 5.5 LangGraph nodes

3 nodes, 1 conditional edge:

```
classify_intent → (policy_question?) → retrieve_and_answer → END
                → (general_question?) → direct_answer       → END
```

State is a plain `dict` (`SupportState`) carrying `query`, `intent`, `retrieved_ids`, `retrieved_docs`, `answer`, `sources`, `confidence`.

### 5.6 Prompt template

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

### 5.7 Structured output

`AskResponse` (Pydantic): `answer: str`, `sources: list[str]`, `confidence: float` (0–1). Mock mode fills this in directly from code. Real-LLM mode validates the raw JSON (`validate_structured_response`) and retries up to 2 times with corrective feedback (`call_llm_with_retry`) before returning a marked error.

### 5.8 Example calls

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

### 5.9 Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

Serves the same `/ask` endpoint on `http://localhost:7860`, `MOCK_LLM=1` by default.

### 5.10 Optional: real LLM mode

```bash
# .env
MOCK_LLM=0
GROQ_API_KEY=your_key_here
```

Free key from console.groq.com. Not required for grading — mock mode is the baseline.

---

## 6. Installing everything at once (optional)

Each module was built and graded independently with its own small dependency set, and that's still the recommended path — it keeps every environment lean and avoids any chance of version conflicts between, say, `scikit-learn` and `chromadb`.

If you'd rather set up a single environment for the whole repo instead (e.g. to poke around all three modules without recreating venvs), there's a consolidated `requirements.txt` at the repo root:

```bash
python -m venv .venv
pip install -r requirements.txt
```

This installs the union of what all three modules need. It's a convenience option, not what grading assumes — the per-module steps in sections 3–5 above are the primary path.

---

## 7. Notes

- `books.db` (Module 1) and `titanic.csv` / `titanic_full_pipeline.joblib` (Module 2) are generated by running the respective scripts/notebooks — they aren't meant to be edited by hand.
- No secrets are committed anywhere in the repo. The only optional credential (`GROQ_API_KEY`) goes in a local `.env` file, which is git-ignored.
