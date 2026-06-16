import os
import re
import requests
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_URL = "http://jsbach.es/bbdd/"
# All five category index pages covering the complete BWV catalogue 1-1200+
INDEX_PAGES = ["index01.htm", "index02.htm", "index03.htm", "index04.htm", "index05.htm"]
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

def get_subpages():
    """Fetches all 5 index pages and extracts all subpage links."""
    all_subpages = set()
    
    for idx_page in INDEX_PAGES:
        idx_url = urljoin(BASE_URL, idx_page)
        print(f"Fetching index page: {idx_url}")
        try:
            response = requests.get(idx_url, timeout=15)
            response.raise_for_status()
            html = response.text
            
            # Robust case-insensitive search for htm/html href links
            links = re.findall(r'(?i)href\s*=\s*[\'"]?([^\'">]+\.htm[l]?)[\'"]?', html)
            
            # Filter to get subpages belonging to categories index01_ to index05_
            prefixes = ["index01_", "index02_", "index03_", "index04_", "index05_"]
            subpage_paths = [p for p in links if any(prefix in p for prefix in prefixes)]
            
            # Build absolute URLs
            for p in subpage_paths:
                all_subpages.add(urljoin(BASE_URL, p))
        except Exception as e:
            print(f"Warning: Failed to fetch index page {idx_url}: {e}")
            
    subpages = sorted(list(all_subpages))
    print(f"Found {len(subpages)} total subpages to scan across all categories.")
    return subpages

def scan_subpage_for_midis(subpage_url):
    """Scans a single subpage and returns all MIDI links found."""
    try:
        response = requests.get(subpage_url, timeout=12)
        if response.status_code != 200:
            return []
        html = response.text
        
        # Robust case-insensitive search for midi href links
        midi_paths = re.findall(r'(?i)href\s*=\s*[\'"]?([^\'">]+\.mid[i]?)[\'"]?', html)
        
        # Resolve to absolute URLs
        midi_urls = [urljoin(BASE_URL, p) for p in midi_paths]
        return midi_urls
    except Exception:
        # Ignore individual page errors to continue scraping
        return []

def download_midi(midi_url):
    """Downloads a single MIDI file to the output directory."""
    try:
        # Create a safe, flat file name based on URL structure
        file_name = midi_url.split("/")[-1]
        out_path = os.path.join(OUTPUT_DIR, file_name)
        
        # If file already exists, don't download it again
        if os.path.exists(out_path):
            return True
            
        response = requests.get(midi_url, timeout=20)
        if response.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(response.content)
            return True
    except Exception:
        pass
    return False

def scrape_all():
    """Main scraping orchestrator."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Scan for all subpages across all categories (BWV 1 to 1200+)
    subpages = get_subpages()
    if not subpages:
        print("Failed to find any subpages.")
        return
        
    # Step 2: Scan all subpages in parallel to gather MIDI URLs
    midi_urls = set()
    print("Scanning subpages for MIDI links...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(scan_subpage_for_midis, url): url for url in subpages}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning pages"):
            urls = future.result()
            midi_urls.update(urls)
            
    print(f"Discovered {len(midi_urls)} total MIDI file URLs.")
    
    if not midi_urls:
        print("No MIDI files found to download.")
        return
        
    # Step 3: Download all MIDI files in parallel
    print(f"Downloading MIDI files to {OUTPUT_DIR}...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_midi, url): url for url in midi_urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading MIDIs"):
            if future.result():
                success_count += 1
                
    print(f"\nCompleted! Successfully downloaded {success_count} MIDI files.")

if __name__ == "__main__":
    scrape_all()
