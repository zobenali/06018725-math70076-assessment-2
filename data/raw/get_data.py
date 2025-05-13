import os
import re
import json
import argparse
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from urllib.parse import quote

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

URL = 'https://www.wikiart.org/fr'
HEADERS_AJAX = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest"
}



def get_artist_info(artist_name):
    options = Options()
    options.headless = True 
    driver = webdriver.Chrome(options=options)

    url = f'{URL}/{artist_name}'
    print(f'Fetching artist info from {url}')
    driver.get(url)

    time.sleep(3)

    birth_year = 'unknown'
    movement = 'unknown'

    try:
        # Extrait date de naissance
        birth_elem = driver.find_element(By.XPATH, '//span[@itemprop="birthDate"]')
        match = re.search(r'\b(1[5-9]\d{2}|20[0-2]\d)\b', birth_elem.text)
        if match:
            birth_year = match.group()
    except Exception:
        pass

    try:
        # Extrait mouvement
        movement_elem = driver.find_element(By.XPATH, '//a[contains(@href, "/fr/artists-by-art-movement/")]')
        movement = movement_elem.text.strip()
    except Exception:
        pass

    driver.quit()
    print(f"Birth year: {birth_year}, Movement: {movement}")
    return birth_year, movement


def get_paintings(artist_name, movement, n_max):
    movement_link = movement.lower().replace(' ', '-')
    code = quote(movement_link)
    print(f'Style{code}')

    url = f"{URL}/{artist_name}/all-works#!#filterName:Style_{code},resultType:masonry"
    print(f'Fetching paintings from {url}')

    ajax_url = f"{URL}/ajax/ArtistAllPaintingsByFilter?json=2&artistUrl={artist_name}&filter=style:{code}"
    res = requests.get(ajax_url, headers=HEADERS_AJAX)
    data = res.json()

    links = []
    for item in data.get('Data', [])[:n_max]:
        img_url = item.get('image')
        if img_url:
            links.append(img_url)
    
    return links

def download_image(urls, artist_name, path):
    os.makedirs(path, exist_ok=True)
    for i, url in enumerate(tqdm(urls, desc=f"Downloading images for {artist_name}")):
        # Check for exceptions
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.save(os.path.join(path, f"{artist_name}_{i}.jpg"))
        except Exception as e:
            print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download paintings from WikiArt.")
    parser.add_argument("--artists", nargs='+', type=str, help="List of artists to download format : name-surname")
    parser.add_argument("--limit", type=int, help="Maximum number of paintings to download")
    parser.add_argument("--out", type=str, default="data/raw/images", help="Path to save the images")
    
    args = parser.parse_args()

    # Create a dictionnary to save artists' movements
    painter_movements = {} 

    for artist in args.artists:
        print(f"Processing {artist}")
        birth_year, movement = get_artist_info(artist)
        last_name = artist.split('-')[-1]

        folder_name = f"{last_name} - {artist}_{birth_year}"
        path = os.path.join(args.out, folder_name)
        
        paintings = get_paintings(artist, movement, args.limit)
        download_image(paintings, artist, path)
        painter_movements[artist] = movement

    dict_path = os.path.join(args.out, 'painter_movements.json')
    with open(dict_path, 'w') as f:
        json.dump(painter_movements, f, indent=4, ensure_ascii=False)
        
    print(f"Done")
