import json
import os
import requests
from bs4 import BeautifulSoup

DATASETS_PATH = r"/Users/aryavinayak/Desktop/hackathon/pharma_assistant/datasets"
DATASETS_MICROLABS_USA = os.path.join(DATASETS_PATH, "microlabs_usa")

# Ensure the output directory exists
os.makedirs(DATASETS_MICROLABS_USA, exist_ok=True)

URLS = {
    "Acetazolamide Extended-Release Capsules": "https://www.microlabsusa.com/products/acetazolamide-extended-release-capsules/",
    "Amlodipine Besylate and Olmesartan Medoxomil Tablets": "https://www.microlabsusa.com/products/amlodipine-besylate-and-olmesartan-medoxomil-tablets/",
    "Amoxicillin and Clavulanate Potassium for Oral Suspension, USP": "https://www.microlabsusa.com/products/amoxicillin-and-clavulanate-potassium-for-oral-suspension-usp/",
    "Amoxicillin and Clavulanate Potassium Tablets, USP": "https://www.microlabsusa.com/products/amoxicillin-and-clavulanate-potassium-tablets-usp/",
    "Amoxicillin Capsules, USP": "https://www.microlabsusa.com/products/amoxicillin-capsules-usp/",
    "Aspirin and Extended-Release Dipyridamole Capsules": "https://www.microlabsusa.com/products/aspirin-and-extended-release-dipyridamole-capsules/",
    "Atorvastatin Calcium Tablets": "https://www.microlabsusa.com/products/atorvastatin-calcium-tablets/",
    "Bimatoprost Ophthalmic Solution": "https://www.microlabsusa.com/products/bimatoprost-ophthalmic-solution/",
    "Celecoxib capsules": "https://www.microlabsusa.com/products/celecoxib-capsules/",
    "Chlordiazepoxide Hydrochloride and Clidinium Bromide Capsules, USP": "https://www.microlabsusa.com/products/chlordiazepoxide-hydrochloride-and-clidinium-bromide-capsules-usp/",
    "Clindamycin Hydrochloride Capsules, USP": "https://www.microlabsusa.com/products/clindamycin-hydrochloride-capsules-usp/",
    "Clobazam Tablets": "https://www.microlabsusa.com/products/clobazam-tablets/",
    "Clobetasol Propionate Topical Solution, USP": "https://www.microlabsusa.com/products/clobetasol-propionate-topical-solution-usp/",
    "Clomipramine Hydrochloride Capsules, USP": "https://www.microlabsusa.com/products/clomipramine-hydrochloride-capsules-usp/",
    "Cromolyn Sodium Inhalation Solution, USP": "https://www.microlabsusa.com/products/cromolyn-sodium-inhalation-solution-usp/",
    "Cromolyn Sodium Oral Solution": "https://www.microlabsusa.com/products/cromolyn-sodium-oral-solution/",
    "Dalfampridine Extended-Release Tablets": "https://www.microlabsusa.com/products/dalfampridine-extended-release-tablets/",
    "Diclofenac Sodium and Misoprostol Delayed-Release Tablets, USP": "https://www.microlabsusa.com/products/diclofenac-sodium-and-misoprostol-delayed-release-tablets-usp/",
    "Dorzolamide HCl and Timolol Maleate Ophthalmic Solution, USP": "https://www.microlabsusa.com/products/dorzolamide-hcl-and-timolol-maleate-ophthalmic-solution-usp/",
    "Dorzolamide HCl and Timolol Maleate Ophthalmic Solution, USP (Preservative Free)": "https://www.microlabsusa.com/products/dorzolamide-hcl-and-timolol-maleate-ophthalmic-solution-usp-preservative-free/",
    "Dorzolamide HCl Ophthalmic Solution, USP": "https://www.microlabsusa.com/products/dorzolamide-hcl-ophthalmic-solution-usp/",
    "Famotidine for Oral Suspension, USP": "https://www.microlabsusa.com/products/famotidine-for-oral-suspension-usp/",
    "Fenofibric Acid Delayed-Release Capsules": "https://www.microlabsusa.com/products/fenofibric-acid-delayed-release-capsules/",
    "Formoterol Fumarate Inhalation Solution": "https://www.microlabsusa.com/products/formoterol-fumarate-inhalation-solution/",
    "Glimepiride Tablets, USP": "https://www.microlabsusa.com/products/glimepiride-tablets-usp/",
    "Ketorolac Tromethamine Ophthalmic Solution": "https://www.microlabsusa.com/products/ketorolac-tromethamine-ophthalmic-solution/",
    "Levocetirizine Dihydrochloride Tablets, USP": "https://www.microlabsusa.com/products/levocetirizine-dihydrochloride-tablets-usp/",
    "Mefenamic Acid Capsules, USP": "https://www.microlabsusa.com/products/mefenamic-acid-capsules-usp/",
    "Metformin Hydrochloride Extended-Release Tablets, USP": "https://www.microlabsusa.com/products/metformin-hydrochloride-extended-release-tablets-usp/",
    "Metformin Hydrochloride Oral Solution": "https://www.microlabsusa.com/products/metformin-hydrochloride-oral-solution/",
    "Methenamine Hippurate Tablets, USP": "https://www.microlabsusa.com/products/methenamine-hippurate-tablets-usp/",
    "Olmesartan Medoxomil Tablets, USP": "https://www.microlabsusa.com/products/olmesartan-medoxomil-tablets-usp/",
    "Piroxicam Capsules, USP": "https://www.microlabsusa.com/products/piroxicam-capsules-usp/",
    "Ramelteon Tablets": "https://www.microlabsusa.com/products/ramelteon-tablets/",
    "Ranolazine Extended-Release Tablets": "https://www.microlabsusa.com/products/ranolazine-extended-release-tablets/",
    "Rasagiline Tablets": "https://www.microlabsusa.com/products/rasagiline-tablets/",
    "Roflumilast Tablets": "https://www.microlabsusa.com/products/roflumilast-tablets/",
    "Rufinamide Tablets, USP": "https://www.microlabsusa.com/products/rufinamide-tablets-usp/",
    "Tafluprost Opthalmic Solution": "https://www.microlabsusa.com/products/tafluprost-opthalmic-solution/",
    "Telmisartan Tablets, USP": "https://www.microlabsusa.com/products/telmisartan-tablets-usp/",
    "Timolol Maleate Ophthalmic Solution, USP": "https://www.microlabsusa.com/products/timolol-maleate-ophthalmic-solution-usp/",
    "Timolol Maleate Ophthalmic Solution, USP (Preservative-Free)": "https://www.microlabsusa.com/products/timolol-maleate-ophthalmic-solution-usp-preservative-free/",
    "Tobramycin Inhalation Solution, USP": "https://www.microlabsusa.com/products/tobramycin-inhalation-solution-usp/",
    "Travoprost Ophthalmic Solution, USP": "https://www.microlabsusa.com/products/travoprost-ophthalmic-solution-usp/"
}

def setup_prescribing_info_urls(urls_map):
    """
    Retrieve Prescribing Information URLs for each product.
    """
    updated_urls = {}

    for key, value in urls_map.items():
        print(f"\n[PROCESSING] Product: {key}")
        print(f"Fetching product page: {value}")
        
        try:
            # Fetch product page
            data = requests.get(value, timeout=10)
            data.raise_for_status()  # Raise an exception for bad status codes
            
            soup = BeautifulSoup(data.text, "html.parser")
            h2 = soup.findAll("h2")  # Look for "Prescribing Information" in h2 tags
            
            got_prescribing_info = False
            
            for h2_item in h2:
                txt = h2_item.get_text()
                if txt and txt.strip().lower() == "prescribing information":
                    child_url = h2_item.findAll("a")
                    if child_url:
                        href = child_url[0].get("href")
                        print(f"Found Prescribing Information URL: {href}")
                        
                        # Fetch Prescribing Information page
                        try:
                            html = requests.get(href, timeout=10)
                            html.raise_for_status()
                            
                            prescribing_soup = BeautifulSoup(html.text, "html.parser")
                            
                            updated_urls[key] = {
                                "product_url": value,
                                "prescribing_info_url": href,
                                "prescribing_soup": prescribing_soup
                            }
                            
                            got_prescribing_info = True
                            print("[SUCCESS] Retrieved Prescribing Information")
                            break
                        
                        except requests.RequestException as e:
                            print(f"[ERROR] Failed to fetch Prescribing Information page: {e}")
            
            if not got_prescribing_info:
                print(f"[WARNING] No Prescribing Information found for {key}")
        
        except requests.RequestException as e:
            print(f"[ERROR] Failed to fetch product page: {e}")
    
    print(f"\n[SUMMARY] Found Prescribing Information for {len(updated_urls)} out of {len(urls_map)} products")
    return updated_urls

def get_text_below_anchor_with_special_handling(a_tag):
    """
    Extract text and handle special elements like tables and images.
    """
    result_text = []

    for sibling in a_tag.find_next_siblings():
        if sibling.name == "div":
            for child in sibling.children:
                if child.name is not None:
                    if child.name.lower() == "table":
                        table_content = []
                        rows = sibling.find_all('tr')
                        for row in rows:
                            cells = [cell.get_text(strip=False) for cell in row.find_all(['td', 'th'])]
                            table_content.append(" ".join(cells))
                        result_text.append("\n".join(table_content))

                    elif child.name.lower() == "img":
                        img_src = sibling.get('src', 'No src attribute')
                        img_alt = sibling.get('alt', 'No alt text')
                        result_text.append(f"Image: [src={img_src}, alt={img_alt}]")
                    else:
                        result_text.append(sibling.get_text(strip=True))

    return "\n".join(result_text)

def get_all_sections(soup):
    """
    Extract all sections from the Prescribing Information.
    """
    atags = soup.findAll("a")
    info = dict()

    for atag in atags:
        if atag:
            at = atag.get("id")
            if at and at.startswith("anch_dj_dj-dj"):
                txt = get_text_below_anchor_with_special_handling(atag)
                info[atag.get_text()] = txt
    return info

def process_prescribing_soup(name, soup):
    """
    Process the Prescribing Information soup and extract relevant content.
    """
    print(f"\n[PROCESSING] Extracting sections for {name}")
    results = get_all_sections(soup)
    results["product_name"] = name
    print(f"[SUCCESS] Extracted {len(results)} sections")
    return results

def create_dataset_file(pth, result):
    """
    Create a JSON file for the product's Prescribing Information.
    """
    try:
        fname = os.path.join(pth, result["product_name"] + ".json")
        print(f"\n[SAVING] Saving data to {fname}")
        
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        
        print("[SUCCESS] File saved successfully")
    
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")

def main():
    print("[START] MicroLabs USA Prescribing Information Scraper")
    
    try:
        modified_urls = setup_prescribing_info_urls(URLS)
        
        for k, v in modified_urls.items():
            try:
                results = process_prescribing_soup(k, v["prescribing_soup"])
                create_dataset_file(DATASETS_MICROLABS_USA, results)
            except Exception as e:
                print(f"[ERROR] Processing failed for {k}: {e}")
        
        print("\n[COMPLETE] Scraping process finished")
    
    except Exception as e:
        print(f"[FATAL ERROR] Scraping process failed: {e}")

if __name__ == '__main__':
    main()