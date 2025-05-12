import requests
from bs4 import BeautifulSoup
import time
import json
import csv
import random
import os

BASE_URL = "https://www.nhs.uk"
START_URL = f"{BASE_URL}/conditions/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

START_INDEX = 800
MAX_CONDITIONS = 950

def get_condition_links():
    response = requests.get(START_URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    panels = soup.find_all("ul", class_="nhsuk-list nhsuk-list--border")
    links = []

    for panel in panels:
        for a in panel.find_all("a", href=True):
            href = a['href']
            if href.startswith("/conditions/"):
                full_link = BASE_URL + href
                if full_link not in links:
                    links.append(full_link)

    return links[:MAX_CONDITIONS]

# We dont use the description of the disease anymore, but we keep the function for future use
def extract_description(soup):
    main_container = soup.find("main")
    if not main_container:
        main_container = soup.find("div", class_="nhsuk-width-container")

    if main_container:
        paragraphs = main_container.find_all("p", recursive=True)
        desc_paragraphs = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text.split()) > 4:
                desc_paragraphs.append(text)
            if len(desc_paragraphs) >= 2:
                break
        if desc_paragraphs:
            return "\n".join(desc_paragraphs).strip()

    return "N/A"


def extract_symptoms(soup, url, visited_urls=None):
    if visited_urls is None:
        visited_urls = set()

    normalized_url = url.rstrip("/")
    if normalized_url in visited_urls:
        print(f"Already visited {normalized_url}.")
        return "N/A"
    visited_urls.add(normalized_url)

    def extract_from_soup(soup_obj):
        content = []
        headings = soup_obj.find_all(["h2", "h3"])
        for heading in headings:
            heading_text = heading.get_text(strip=True).lower()
            #What are the signs of dyslexia?
            #Symptoms of acute pancreatitis
            if "symptom" in heading_text or "sign" in heading_text:
                for sibling in heading.find_next_siblings():
                    if sibling.name in ["h2", "h3"]:
                        break
                    if sibling.name == "p":
                        content.append(sibling.get_text(strip=True))
                    elif sibling.name == "ul":
                        for li in sibling.find_all("li"):
                            content.append("- " + li.get_text(strip=True))
                if content:
                    return "\n".join(content).strip()

        for p in soup_obj.find_all("p"):
            p_text = p.get_text(strip=True).lower()
            if "symptom" in p_text:
                next_tag = p.find_next_sibling()
                if next_tag and next_tag.name == "ul":
                    for li in next_tag.find_all("li"):
                        content.append("- " + li.get_text(strip=True))
                    return "\n".join(content).strip()

        return None

    result = extract_from_soup(soup)
    if result:
        return result

    current_path = url.replace(BASE_URL + "/conditions/", "").strip("/").split("/")[0]
    candidate_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        full_link = BASE_URL + href if href.startswith("/conditions/") else href
        if "symptom" in href and "/conditions/" in href and current_path in href:
            full_link = full_link.rstrip("/")
            if full_link not in visited_urls and full_link not in candidate_links:
                candidate_links.append(full_link)

    for link in candidate_links:
        try:
            print(f"Trying related symptom page: {link}")
            time.sleep(random.uniform(3.5, 10.5))
            resp = requests.get(link, headers=HEADERS)
            if resp.status_code == 200:
                new_soup = BeautifulSoup(resp.text, "html.parser")
                result = extract_symptoms(new_soup, link, visited_urls)
                if result and result != "N/A":
                    return result
        except Exception as e:
            print(f"Failed to access {link}: {e}")

    return "N/A"




def scrape_condition(url):
    print(f" Scraping: {url}")
    try:
        time.sleep(random.uniform(5.5, 10.5))
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print(f" Page not found: {url}")
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("h1").text.strip()
    except Exception as e:
        print(f" Error at title: {e}")
        return None

    return {
        "disease": title,
        "description": extract_description(soup),
        "symptoms": extract_symptoms(soup, url, set())
    }



def load_existing_data(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


all_data = []

print(" Extracting links...")
links = get_condition_links()
print(f"Found {len(links)} links.")

links_to_scrape = links[START_INDEX:]

for idx, link in enumerate(links_to_scrape, start=START_INDEX + 1):
    try:
        data = scrape_condition(link)
        if data:
            all_data.append(data)
        print(f" {idx}/{len(links)} complete.")
        time.sleep(random.uniform(2, 8))
    except Exception as e:
        print(f" Erorr at {link}: {e}")



print(f"\nDiseases length: {len(all_data)}")

existing_data = load_existing_data("data/boli_nhs.json")
combined_data = existing_data + all_data

seen = set()
final_data = []
for item in combined_data:
    name = item.get("disease", "").strip().lower()
    if name and name not in seen:
        final_data.append(item)
        seen.add(name)


with open("data/boli_nhs.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)


with open("data/boli_nhs.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["disease", "description", "symptoms"])
    writer.writeheader()
    for entry in final_data:
        writer.writerow(entry)

print(f"\n Final save in boli_nhs.json and boli_nhs.csv — total: {len(final_data)} diseases")
