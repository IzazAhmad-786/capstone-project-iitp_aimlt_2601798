# Data Pipeline — Module 1 (`/data_pipeline`)

This project scrapes book data from **books.toscrape.com**, cleans it, converts the price to INR using a fixed rate, stores it in a normalized SQLite database, and runs SQL + pandas queries on it.

---

## 1. Data Source

- Website: https://books.toscrape.com/
- This is a public site made only for scraping practice. No login or API key is needed.
- We scrape the first **5 categories** from the site (can be changed using `NUM_CATEGORIES` in the script).
- For each book we collect: `title`, `price`, `star_rating`, `availability`, `category`.

---

## 2. Requirements

Install these Python packages before running:

```
pip install requests beautifulsoup4 pandas
```

(Python's `sqlite3` module comes built-in, no install needed.)

---

## 3. How to Run

```
python data_pipeline.py
```

This single command will:

1. Scrape the book data from the website.
2. Clean and convert the data.
3. Create/rebuild the SQLite database (`books.db`).
4. Load the cleaned data into the database.
5. Run 5 SQL queries and print the results.
6. Run the same 2 queries again using `pd.read_sql`.
7. Reproduce the JOIN query using `pd.merge` (no SQL) and check that both results match.

To save the output into a file (recommended for submission), run:

```
python data_pipeline.py > query_output.txt
```

---

## 4. Data Cleaning Decisions

| Field | Raw Format | Cleaned Format | How |
|---|---|---|---|
| `price` | `"£51.77"` | `price_gbp` (float) | Removed the `£` symbol and a stray encoding character (`Â`), then converted to a number. |
| `star_rating` | Text class like `"Three"` | `rating` (int 1–5) | Mapped word → number using a dictionary (`One` → 1, ... `Five` → 5). |
| `availability` | `"In stock (22 available)"` | `in_stock` (True/False) | Checked if the text contains `"in stock"`. |

**Missing / bad values:** If `price_gbp` or `rating` could not be parsed for some row, we used **median imputation** (filled the missing value with the median of that column) instead of dropping the row. We chose imputation over dropping rows because losing a full book record was a bigger loss than approximating one number, and the dataset is small (needs ≥60 rows), so we did not want to drop the required minimum by accident.

---

## 5. Currency Conversion

We use a **fixed, project-defined rate**:

```
1 GBP = 105.50 INR
```

This is a constant used only for this assignment. It is **not** a live or historical market rate, so no API call or date lookup is needed.

```
price_inr = price_gbp * 105.50
```

---

## 6. Database Schema

SQLite database file: `books.db` (regenerated automatically each time the script runs).

**Table: `categories`**
| Column | Type |
|---|---|
| category_id | INTEGER PRIMARY KEY |
| category_name | TEXT UNIQUE |

**Table: `books`**
| Column | Type |
|---|---|
| book_id | INTEGER PRIMARY KEY |
| title | TEXT |
| price_gbp | REAL |
| price_inr | REAL |
| rating | INTEGER |
| in_stock | BOOLEAN |
| category_id | INTEGER (FOREIGN KEY → categories.category_id) |

Indexes were added on `category_name`, `category_id`, `price_inr`, `rating`, and `in_stock` for faster queries.

---

## 7. SQL Queries

All 5 queries are in `run_queries()` inside `data_pipeline.py`. They are also printed when you run the script.

| # | Purpose | SQL feature shown |
|---|---|---|
| Q1 | In-stock books over ₹300 | SELECT / WHERE |
| Q2 | 10 most expensive books | ORDER BY / LIMIT |
| Q3 | Distinct category names | DISTINCT |
| Q4 | 4–5 star books priced ₹500–₹2000 | IN / BETWEEN |
| Q5 | Top 10 highest-rated books per category | JOIN (with window function) |

---

## 8. Pandas Verification

- `pd.read_sql` was used to re-run **Q1** and **Q2** directly into DataFrames.
- The **Q5 (JOIN)** result was reproduced **without SQL**, using `pd.merge()` on the in-memory `books` and `categories` DataFrames, plus a manual ranking step (`groupby` + `cumcount`) to copy what `ROW_NUMBER()` did in SQL.
- The script compares both results with `.equals()` and prints:
  ```
  SQL result matches pandas merge result: True
  ```

---

## 9. Files in this Module

- `data_pipeline.py` — full pipeline (scrape → clean → convert → load → query)
- `books.db` — SQLite database (auto-created by the script)
- `output.txt` — saved output of all queries (create this by running the script with `>`)
- `README.md` — this file