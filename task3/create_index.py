import json
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

from task2.tokenize_lemmatize import PAGES_DIR, process_text

OUTPUT_DIR = Path(__file__).parent / ''
INDEX_FILE = 'inverted_index.json'

def extract_text_from_html(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='windows-1251', errors='replace') as f:
            html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    text_parts = []
    for z_tag in soup.find_all('z'):
        for o_tag in z_tag.find_all('o'):
            parent = o_tag.parent
            o_tag.decompose()
            text_parts.append(parent.get_text(separator=' ', strip=True))
    
    return ' '.join(text_parts)


def create_inverted_index():
    print(f"Загрузка документов из {PAGES_DIR}...")

    html_files = sorted(PAGES_DIR.glob('*.html'), key=lambda x: int(x.stem) if x.stem.isdigit() else 999999)
    if html_files:
        print(f"Найдено {len(html_files)} HTML файлов")

    print(f"Обработка {len(html_files)} файлов...")
    
    global_index = defaultdict(dict)
    
    for i, filepath in enumerate(html_files, 1):
        doc_id = filepath.stem
        print(f"[{i}/{len(html_files)}] Обработка файла {doc_id}...", end='\r')
        
        text = extract_text_from_html(filepath)
        
        if not text:
            continue
        
        _, _, doc_index = process_text(text)
        
        for lemma, positions in doc_index.items():
            global_index[lemma][doc_id] = sorted(list(set(positions)))
    
    print(f"\n\nСоздан инвертированный индекс для {len(global_index)} уникальных лемм")
    print(f"Обработано {len(html_files)} документов")
    
    total_entries = sum(len(docs) for docs in global_index.values())
    print(f"Всего записей в индексе: {total_entries}")
    
    return dict(global_index)


def save_index(index, output_path):
    print(f"Сохранение индекса в {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    index = create_inverted_index()
    save_index(index, INDEX_FILE)
    
    print(f"\nИнвертированный индекс создан и сохранен в:")
    print(f"  {INDEX_FILE}")

if __name__ == '__main__':
    main()
