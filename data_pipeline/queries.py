import sqlite3 as sql
import pandas as pd

DB_FILE_PATH = "output/books.db"

def execute_query(conn, title, query):
    print(f"---{title}---")
    print(query.strip())

    df = pd.read_sql_query(query, conn)
    print(df)
    print("---"*30)
    return df

def run_queries():
    conn = sql.connect(DB_FILE_PATH)
    # 1. SELECT / WHERE
    execute_query(conn,"Q1: In-stock books over 300 INR", "SELECT title, price_inr FROM books WHERE in_stock = 1 AND price_inr > 300;")

    # 2. ORDER BY + LIMIT
    execute_query(conn,"Q2: 10 most expensive books (INR)", "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;")

    # 3. DISTINCT
    execute_query(conn,"Q3: Distinct category names present", "SELECT DISTINCT category_name FROM categories ORDER BY category_name;")

    # 4. IN / BETWEEN
    execute_query(conn,"Q4: Books rated 4 or 5 stars priced between 500 and 2000 INR", "SELECT title, rating, price_inr FROM books WHERE rating IN (4, 5) AND price_inr BETWEEN 500 AND 2000;")

    # 5. JOIN - top 10 highest-rated books per category
    sql_result = execute_query(conn,"Q5: All books joined with category name", 
                  """
                    SELECT category_name, title, rating, price_inr FROM (
                        SELECT c.category_name, b.title, b.rating, b.price_inr,
                        ROW_NUMBER() OVER (
                            PARTITION BY c.category_id ORDER BY b.rating DESC, b.price_inr DESC
                        ) AS rnk
                        FROM books b JOIN categories c ON b.category_id = c.category_id
                    )
                    WHERE rnk <= 10 ORDER BY category_name, rnk
                """)

    # --- pandas: read_sql for two of the above queries ---
    df_q1 = pd.read_sql("SELECT title, price_inr FROM books WHERE in_stock = 1 AND price_inr > 300;", conn)
    df_q2 = pd.read_sql("SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;", conn)

    print("\n--- pd.read_sql Q1 (head) ---")
    print(df_q1.head())
    print("\n--- pd.read_sql Q2 ---")
    print(df_q2)

    # --- pandas: reproduce the JOIN using pd.merge on in-memory DataFrames ---
    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)

    conn.close()

    merged = pd.merge(books_df, categories_df, on="category_id")
    # kind="mergesort" is stable, so books tied on rating AND price_inr keep
    # their original row order instead of an arbitrary quicksort order —
    # otherwise rnk (and the equality check below) can flip between runs.
    merged["rnk"] = (
            merged.sort_values(["rating", "price_inr"], ascending=[False, False], kind="mergesort")
            .groupby("category_id")
            .cumcount() + 1
        )
    pandas_result = (
        merged[merged["rnk"] <= 10]
            .sort_values(["category_name", "rnk"])[["category_name", "title", "rating", "price_inr"]]
            .reset_index(drop=True)
        )
    print("\n--- Same JOIN result via pd.merge (no SQL) ---")
    print(pandas_result)

    match = sql_result.reset_index(drop=True).equals(pandas_result)
    print(f"\nSQL result matches pandas merge result: {match}")


if __name__ == "__main__":
    run_queries()