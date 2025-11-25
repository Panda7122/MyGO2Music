import sys
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import time
import os
import re
import base64
from urllib.parse import urljoin, urlparse

#!/usr/bin/env python3
"""
get_mygo_image.py
Fetch HTML from https://mygo.miyago9267.com/ using requests + BeautifulSoup and save it locally.
"""

URL = "https://mygo.miyago9267.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mygo-fetcher/1.0)"}
TIMEOUT = 10

def main():
    os.makedirs('./mygo_image/', exist_ok=True)
    chrome_opts = Options()
    # run headless; depending on Chrome/selenium version you may need "--headless" instead
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_opts)
    driver.set_page_load_timeout(TIMEOUT)
    driver.get(URL)
    # wait until the document is fully loaded (useful for JS-rendered pages)
    WebDriverWait(driver, TIMEOUT).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    time.sleep(1)
    try:
        btn = WebDriverWait(driver, TIMEOUT).until(
            lambda d: d.find_element("xpath", "//button[normalize-space(.)='×']")
        )
        btn.click()
        time.sleep(0.5)
    except Exception as e:
        print("Could not find/click close button:", e)
    while True:
        last_height = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
    soup = BeautifulSoup(driver.page_source, "html.parser")
    rows = soup.find_all("div", class_="image-row")
    print(f"Found {len(rows)} image-row div(s)")
    for i, row in enumerate(rows, 1):
        imgs = row.find_all("img")
        print(f"row {i}: {len(imgs)} img(s)")
        for img_html in imgs:
            src = img_html.get("src")
            if not src:
                continue
            filename = src.split('/')[-1].split('?')[0]
            print(filename)
            # make absolute URL if needed
            # src_url = urljoin(URL, src)

            try:
                resp = requests.get(src, headers=HEADERS, timeout=TIMEOUT)
                resp.raise_for_status()
                # determine extension from Content-Type or URL path
                content_type = resp.headers.get("Content-Type", "")
                ext = None
                if "image/" in content_type:
                    ext = content_type.split("image/")[-1].split(";")[0]
                    if ext == "jpeg":
                        ext = "jpg"
                if not ext:
                    path = urlparse(src).path
                    _, ext = os.path.splitext(path)
                    ext = ext.lstrip(".") or "jpg"
                save_path = os.path.join('./mygo_image/', filename)
                with open(save_path, "wb") as f:
                    f.write(resp.content)
                print(f"Saved {save_path} from {src}")
            except Exception as e:
                print(f"Failed to download {src}: {e}")
if __name__ == "__main__":
    main()