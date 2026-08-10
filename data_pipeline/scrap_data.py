import os
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"
OUTPUT_DIR = "output/"
OUTPUT_FILE_NAME = OUTPUT_DIR+"raw_books.csv"
NUM_CATEGORIES = 5
    
def get_categories():
    resp = requests.get(urljoin(BASE_URL, "index.html") )
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")

    categories = []
    for link in soup.select("div.side_categories ul li ul li a"):
        name = link.get_text(strip=True)
        url = urljoin(BASE_URL, link["href"])
        categories.append((name, url))
    return categories


def scrape_category(name, url):
    books = []
    while url:
        resp = requests.get(url)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")

        for pod in soup.select("article.product_pod"):
            title = pod.h3.a["title"]
            price = pod.select_one("p.price_color").get_text(strip=True)
            rating = pod.select_one("p.star-rating")["class"][1]  # e.g. "Three"
            availability = pod.select_one("p.instock.availability").get_text(strip=True)

            books.append({
                "title": title,
                "price": price,
                "star_rating": rating,
                "availability": availability,
                "category": name,
            })

        next_link = soup.select_one("ul.pager li.next a")
        # Resolve relative to the CURRENT page url, not string-concatenated,
        # since next_link["href"] (e.g. "page-2.html") is relative to url's
        # directory, not a suffix to append to it.
        url = urljoin(url, next_link["href"]) if next_link else None

    return books

def create_output_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def scrap_data():
    all_books = []
    categories = get_categories()[:NUM_CATEGORIES]

    for name, url in categories:
        books = scrape_category(name, url)
        print(f"{name}: {len(books)} books")
        all_books.extend(books)

    print(f"Total: {len(all_books)} books across {len(categories)} categories")

    create_output_directory()

    with open(OUTPUT_FILE_NAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "price", "star_rating", "availability", "category"])
        writer.writeheader()
        writer.writerows(all_books)

    print(f"Saved to {OUTPUT_FILE_NAME}")

if __name__ == "__main__":
    scrap_data()