import requests
import pandas as pd
import sqlite3 as sql
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"
NUM_CATEGORIES = 5

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
GBP_TO_INR = 105.50

DB_FILE_PATH = "books.db"
SCHEMA = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
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

CREATE INDEX idx_categories_category_name ON categories(category_name);

CREATE INDEX idx_books_category_id ON books(category_id);
CREATE INDEX idx_books_price_inr ON books(price_inr);
CREATE INDEX idx_books_rating ON books(rating);
CREATE INDEX idx_books_in_stock ON books(in_stock);
"""
    
def get_categories():
    resp = requests.get(BASE_URL + "index.html")
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")

    categories = []
    for link in soup.select("div.side_categories ul li ul li a"):
        name = link.get_text(strip=True)
        url = BASE_URL + link["href"]
        categories.append((name, url))
    return categories


def scrape_books(name, url):
    books = []
    while url:
        resp = requests.get(url)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")

        for pod in soup.select("article.product_pod"):
            title = pod.h3.a["title"]
            price = pod.select_one("p.price_color").get_text(strip=True)
            rating_tag = pod.select_one("p.star-rating")
            rating = rating_tag["class"][1]if rating_tag else None
            availability = pod.select_one("p.instock.availability").get_text(strip=True)

            books.append({
                "title": title,
                "price": price,
                "star_rating": rating,
                "availability": availability,
                "category": name,
            })

        next_link = soup.select_one("ul.pager li.next a")
        url = urljoin(url, next_link["href"]) if next_link else None

    return books

def scrape_data():
    print("Scraping book data from the website.")
    all_books = []
    categories = get_categories()[:NUM_CATEGORIES]

    for name, url in categories:
        books = scrape_books(name, url)
        print(f"{name}: {len(books)} books.")
        all_books.extend(books)

    print(f"Total: {len(all_books)} books across {len(categories)} categories.")
    return pd.DataFrame(all_books)

def clean_data(df):
    print(f"Cleaning data: {len(df)} rows, {df.isna().sum().sum()} missing values.")
    # Strip the currency symbol from price
    df["price_gbp"] = (
        df["price"]
        .str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .str.strip()
    )

    # Convert it to a float column price_gbp
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")

    # Convert the text star rating (One…Five) into an integer column rating (1–5).
    df["rating"] = df["star_rating"].map(RATING_WORDS)

    # Create a boolean column indicating whether the book is in stock
    df["in_stock"] = df["availability"].str.lower().str.contains("in stock", na=False)

    # Median-impute any rows that failed to parse numeric fields
    for col in ["price_gbp", "rating"]:
        n_missing = df[col].isna().sum()
        if n_missing:
            median_val = df[col].median()
            print(f"Imputing {n_missing} missing '{col}' values with median {median_val}.")
            df[col] = df[col].fillna(median_val)

    df["rating"] = df["rating"].round().astype(int)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    cleaned_data = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]

    print(f"Cleaned data: {len(cleaned_data)} rows, {cleaned_data.isna().sum().sum()} missing values.")
    return cleaned_data

def get_connection():
    connection = sql.connect(DB_FILE_PATH)
    print("Connection established successfully!.")
    return connection

def load_data(conn, df):
    print(f"Loading data into database.")

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

def execute_query(conn, title, query):
    print(f"---{title}---")
    print(query.strip())

    df = pd.read_sql_query(query, conn)
    print(df)
    print("---"*30)
    return df

def run_queries(conn):
    print("Running queries on the database.")

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
    print(f"\nSQL result matches pandas merge result: {match}.")

def main():
    # Scrape the data       
    all_books_data = scrape_data()

    #cleaning the data
    cleaned_data = clean_data(all_books_data)

    # Establish a connection to the SQLite database
    conn = get_connection()

    # Load the cleaned data into the database
    load_data(conn, cleaned_data)

    # Run the queries on the database
    run_queries(conn)

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main()