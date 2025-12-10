import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import time
import os
import math
from urllib.parse import urlparse

#!/usr/bin/env python3
"""
get_mygo_image.py
Fetch HTML from https://mygo.miyago9267.com/ using requests + BeautifulSoup and save it locally.
"""

URL = "https://mygo.miyago9267.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mygo-fetcher/1.0)"}
TIMEOUT = 10
api_url = "https://mygo.miyago9267.com/api/v1/images"

def main():
    os.makedirs('./mygo_image/', exist_ok=True)
    # get number of page
    res = requests.get(f"{api_url}?page=1&limit=100&order=id")
    total_pages = res.json()['meta']['totalPages']
    for i in range(1, total_pages+1):
        print(f"page {i}")
        res = requests.get(f"{api_url}?page={i}&limit=100&order=id")
        data=  res.json()['data']
        for img_data in data:
            url = img_data['url']
            ext = os.path.splitext( img_data['filename'])[1]
            filename = f"{img_data['id']}_{img_data['alt']}{ext}"
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            save_path = os.path.join('./mygo_image/', filename)
            with open(save_path, "wb") as f:
                f.write(resp.content)
            print(f"Saved {save_path} from {url}")
if __name__ == "__main__":
    main()