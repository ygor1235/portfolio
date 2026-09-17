import os
import time
import json
import sys
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- INITIAL CONFIGURATIONS ---
output_folder = "data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Global dictionaries to store accumulated data
scraped_items = {"title": [], "price": [], "link": []}
categorized_data = {
    "expensive": {"title": [], "price": [], "link": []},
    "average": {"title": [], "price": [], "link": []},
    "promotion": {"title": [], "price": [], "link": []}
}


# --- FUNCTION 1: ADVANCE PAGE (SELENIUM) ---
def advance_page(driver):
    """Tries to click the 'Next' pagination button. Returns True if successful, False if it fails."""
    try:
        next_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[@data-andes-pagination-control='next']"))
        )
        # Scrolls to the button to guarantee visibility, then triggers click
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
        driver.execute_script("arguments[0].click();", next_button)

        # Waits for the new page to render its HTML content before moving forward
        time.sleep(4)
        return True
    except Exception as error:
        print(f"\n[WARNING] Unable to advance to the next page. Error: {error}", file=sys.stderr)
        return False


# --- FUNCTION 2: DATA EXTRACTION AND CLEANING (BEAUTIFULSOUP) ---
def extract_and_parse_data(html_code):
    """Extracts target products from the rendered HTML and appends them to the global structures."""
    soup = BeautifulSoup(html_code, "html.parser")
    items = soup.find_all("li", class_="ui-search-layout__item")

    print(f"-> Products extracted on this page: {len(items)}")

    for item in items:
        # Extract title attribute
        title_element = item.find("h3", class_="poly-component__title-wrapper")
        title = title_element.text.strip() if title_element else "Title not found"

        # Extract pricing structure
        price_element = item.find("span", class_="andes-money-amount__fraction")
        if price_element:
            # Converts into a clean float string by stripping local visual dot delimiters
            clean_price = price_element.text.strip()
        else:
            continue  # Skips item configurations missing a price block to protect mean calculations

        # Extract target destination link
        link_element = item.find("a", class_="poly-component__title")
        link = link_element["href"] if link_element else "Link not found"

        # Anti-bot defensive handling: filter out dummy or internal testing ads
        if "teste" in title.lower():
            continue
        else:
            scraped_items["title"].append(title)
            scraped_items["price"].append((float(clean_price.replace('.', ''))))
            scraped_items["link"].append(link)


# --- AUTOMATED BROWSER ORCHESTRATION ---
chrome_options = Options()
chrome_options.add_argument("--log-level=3")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Target Search Campaign URL (Mercado Libre Nvidia GeForce)
url = "https://lista.mercadolivre.com.br/nvidia-geforce?matt_tool=35035576&matt_word=Default_URL_MLB&matt_source=google&matt_campaign_id=11617901711&matt_ad_group_id=114154246998&matt_match_type=p&matt_network=g&matt_device=c&matt_creative=479799911484&matt_keyword=nvidia%20geforce%20mercado%20livre&matt_ad_position=&matt_ad_type=&matt_merchant_id=&matt_product_id=&matt_product_partition_id=&matt_target_id=kwd-1019589645006&cq_src=google_ads&cq_cmp=11617901711&cq_net=g&cq_plt=gp&cq_med=&gad_source=1&gad_campaignid=11617901711&gbraid=0AAAAAD93qcBQmrIkpuJ7nbNE2D-NOF5Me&gclid=CjwKCAjw2aPVBhBkEiwA0Cpttxrp8dQZFuV5IUGpZSMrJEyVQLNPBWatmNZOFPvBoQWyrgmLCpMf9hoCbIMQAvD_BwE"

try:
    print("[INFO] Launching automated browser and loading target URL...")
    driver.get(url)
    time.sleep(5)

    total_pages = 10

    # DATA HARVESTING ITERATION LOOP
    for page in range(1, total_pages + 1):
        print(f"\n--- Scanning Page {page} of {total_pages} ---")

        # Executes parsing algorithm using the active driver's snapshot source
        extract_and_parse_data(driver.page_source)

        # Triggers pagination movement rule if boundaries aren't reached
        if page < total_pages:
            success = advance_page(driver)
            if not success:
                print("[INFO] Terminating scan pipeline early: No additional pages available.")
                break

    # --- FINANCIAL DATA ANALYTICS SUBSECTION ---
    if scraped_items["price"]:
        print(scraped_items["price"])

        global_average = np.mean(scraped_items["price"])
        print(f"\n[STATISTICS] Global calculated average price: R$ {global_average:.2f}")

        # Define uma margem de tolerância de 10% para cima e para baixo
        lower_bound = global_average * 0.90
        upper_bound = global_average * 1.10
        print(f"[STATISTICS] Market Price Range (Average): R$ {lower_bound:.2f} to R$ {upper_bound:.2f}")

        for i, price in enumerate(scraped_items["price"]):
            # Classifica o produto usando as margens calculadas
            if price < lower_bound:
                category = "promotion"
            elif price > upper_bound:
                category = "expensive"
            else:
                category = "average"  # Agora os itens que estão na média do mercado vão cair aqui!

            categorized_data[category]["title"].append(scraped_items["title"][i])
            categorized_data[category]["price"].append(scraped_items["price"][i])
            categorized_data[category]["link"].append(scraped_items["link"][i])

        # --- REPO REPORT GENERATION VIA PANDAS ---
        for key in categorized_data.keys():
            if categorized_data[key]["title"]:  # Assures current price tier contains rows
                df = pd.DataFrame(categorized_data[key])
                file_path = os.path.join(output_folder, f"products_{key}.csv")

                # Using utf-8-sig to safely preserve specific local characters when viewed in Excel
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
                print(f"[EXPORT SUCCESS] Saved: {file_path} containing {len(df)} records.")

    else:
        print("\n⚠️ [ALERT] No clean item data entities were gathered during this pipeline run.")

finally:
    driver.quit()
    print("\n[INFO] Automation instance safely shut down.")
