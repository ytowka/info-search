import requests
from pathlib import Path
import time
import re

INDEX_FILE = Path(__file__).parent / 'index.txt'
PAGES_DIR = Path(__file__).parent / 'pages'

REQUEST_TIMEOUT = 30
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2
REQUEST_DELAY = 1


def load_urls():
    urls = []
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^\s*\d+\.\s+(.+?)\s*$', line)
            if match:
                urls.append(match.group(1))
    return urls


def download_page(url, index):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            return response.text
        except (requests.RequestException, requests.Timeout) as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f'Error downloading {url}: {e}')
    return None


def main():
    urls = load_urls()
    PAGES_DIR.mkdir(exist_ok=True)
    
    print(f'Found {len(urls)} URLs')
    print('Downloading pages...\n')
    
    for i, url in enumerate(urls, 1):
        print(f'[{i}/{len(urls)}] Downloading...', end='\r')
        content = download_page(url, i)
        
        if content:
            filepath = PAGES_DIR / f'{i}.html'
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            time.sleep(REQUEST_DELAY)
    
    print(f'\nCompleted. Downloaded {len(urls)} pages to {PAGES_DIR}')


if __name__ == '__main__':
    main()
