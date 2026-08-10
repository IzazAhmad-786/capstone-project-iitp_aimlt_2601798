import pandas as pd

INPUT_FILE_PATH = "output/raw_books.csv"
OUTPUT_FILE_PATH = "output/clean_books.csv"

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

GBP_TO_INR = 105.50

def clean_data():
    df = pd.read_csv(INPUT_FILE_PATH)

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
            print(f"Imputing {n_missing} missing '{col}' values with median {median_val}")
            df[col] = df[col].fillna(median_val)

    df["rating"] = df["rating"].round().astype(int)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    cleaned_data = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]
    cleaned_data.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Wrote {len(cleaned_data)} cleaned rows -> {OUTPUT_FILE_PATH}")
    
if __name__ == "__main__":
    clean_data()