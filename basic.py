import json
import re
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Global structure to hold targeted URLs
validated_links = {"ready_links": [], "absolute_links": []}


def extract_base_links(target_url: str) -> None:
    """Extracts and categorizes anchor links from the target URL."""
    global validated_links

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive"
    }

    try:
        # Explicit timeout to prevent indefinitely hung connections
        response = requests.get(target_url, headers=headers, timeout=10)
        # Automatically raises an exception for 4xx or 5xx status codes
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        print(f"[NETWORK ERROR] Failed to fetch data: {error}", file=sys.stderr)
        return

    # Using 'lxml' parser for maximum processing speed
    soup = BeautifulSoup(response.content, "lxml")
    nav_links = soup.find_all("a")

    if not nav_links:
        print("[WARNING] No anchor elements were located on the page.")
        return

    for item in nav_links:
        link = item.get("href")

        # Defensive quick validation
        if link and link.startswith("https"):
            validated_links["ready_links"].append(link)
        elif link:
            absolute_url = urljoin(target_url, link)
            if absolute_url.startswith("https://www.dicio.com.br/"):
                validated_links["absolute_links"].append(absolute_url)


def text_extractor(url_list: list) -> None:
    """Scrapes, cleans, and tokens text from a limited slice of URLs, then saves to JSON."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive"
    }
    extracted_text = []

    for i, target_url in enumerate(url_list):
        try:
            response = requests.get(target_url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as error:
            print(f"[NETWORK ERROR] Failed to fetch target page: {error}", file=sys.stderr)
            return

        soup = BeautifulSoup(response.content, "lxml")

        # 1. Clean whitespaces, line breaks, carriage returns, and tabs
        cleaned_text = re.sub(r'[\n\r\t]+', ' ', soup.text)

        # 2. Tokenize string by standard spaces
        raw_tokens = cleaned_text.strip().split(" ")

        # 3. Filter out empty strings and isolated hyphens
        filtered_tokens = [token for token in raw_tokens if token.strip() and token != '-']
        extracted_text.append(filtered_tokens)

        # Limit the execution to the first 5 pages (index 0 to 4)
        if i == 4:
            break

    # Save structured clean data to the output file
    with open("absolute_data.json", "w", encoding='utf-8') as output_file:
        json.dump(extracted_text, output_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    BASE_URL = "https://www.dicio.com.br/significado/"
    extract_base_links(BASE_URL)
    text_extractor(validated_links["absolute_links"])
