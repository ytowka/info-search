from pathlib import Path
from bs4 import BeautifulSoup
import re

PAGES_DIR = Path(__file__).parent / 'pages'
OUTPUT_DIR = Path(__file__).parent / 'pages_no_links'


def remove_links_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for link in soup.find_all('a', href=True):
        link.unwrap()
    
    for link in soup.find_all('link', href=True):
        link.decompose()
    
    for script in soup.find_all('script', src=True):
        script.decompose()
    
    for img in soup.find_all('img', src=True):
        img.decompose()
    
    for meta in soup.find_all('meta', content=True):
        content = meta.get('content', '')
        if 'http://' in content or 'https://' in content:
            meta.decompose()
    
    for tag in soup.find_all():
        for attr in list(tag.attrs.keys()):
            value = tag.get(attr, '')
            if isinstance(value, str):
                if value.startswith('http://') or value.startswith('https://'):
                    del tag[attr]
    
    result = str(soup)
    
    result = re.sub(r'<a\s+[^>]*?href\s*=\s*["\'][^"\']*["\'][^>]*>', '', result, flags=re.IGNORECASE)
    result = re.sub(r'</a\s*>', '', result, flags=re.IGNORECASE)
    
    result = re.sub(r'\s*https?://[^\s<>"\']+[\w/]*', '', result)
    
    return result


def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    clean_html = remove_links_from_html(html_content)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(clean_html)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    html_files = sorted(PAGES_DIR.glob('*.html'), key=lambda x: int(x.stem))
    
    print(f'Found {len(html_files)} HTML files')
    print('Processing...\n')
    
    for i, html_file in enumerate(html_files, 1):
        output_path = OUTPUT_DIR / f'{i}.txt'
        process_file(html_file, output_path)
        print(f'[{i}/{len(html_files)}] Processed {html_file.name} -> {output_path.name}', end='\r')
    
    print(f'\nCompleted. Processed {len(html_files)} files to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
