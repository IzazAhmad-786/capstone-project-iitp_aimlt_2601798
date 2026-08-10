import sqlite3 as sql
import pandas as pd

CSV_FILE_PATH = "output/clean_books.csv"
DB_FILE_PATH = "output/books.db"

SCHEMA = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock BOOLEAN NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""

def load_data():
    df = pd.read_csv(CSV_FILE_PATH)

    conn = sql.connect(DB_FILE_PATH)
    print(f"Connected to database: {DB_FILE_PATH}")

    cur = conn.cursor()
    cur.executescript(SCHEMA)

    # Insert categories into the categories table
    categories = pd.DataFrame(sorted(df["category"].unique()), columns=["category_name"])

    categories.to_sql("categories", conn, if_exists="append", index=False)

    print(f"Inserted {len(categories)} categories into the database.")
    conn.commit()

    cat_ids = pd.read_sql("SELECT category_id, category_name FROM categories", conn)

    merged = df.merge(cat_ids, left_on="category", right_on="category_name")

    books = merged[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]]

    books.to_sql("books", conn, if_exists="append", index=False)

    print(f"Inserted {len(books)} books into the database.")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    load_data()