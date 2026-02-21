import sys
import math
import re
from pathlib import Path
from collections import defaultdict
from bs4 import BeautifulSoup
from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger
import pymorphy2


PAGES_DIR = Path(__file__).parent.parent / 'task1' / 'pages'
INPUT_DIR = Path(__file__).parent.parent / 'task2' / 'output'
OUTPUT_DIR = Path(__file__).parent / 'output'

CONJUNCTIONS = {
    'и', 'а', 'но', 'или', 'однако', 'зато', 'если', 'коли', 'когда',
    'лишь', 'только', 'что', 'чтобы', 'как', 'ежели', 'нежели', 'хотя',
    'хоть', 'пусть', 'раз', 'кабы', 'коль', 'дабы', 'ибо', 'ведь',
    'же', 'ли', 'да', 'неужели', 'неужто', 'также', 'тоже', 'причем',
    'причём', 'словно', 'будто', 'что-то', 'кое-что'
}
PREPOSITIONS = {
    'в', 'на', 'с', 'к', 'по', 'из', 'за', 'от', 'без', 'через',
    'для', 'при', 'о', 'об', 'у', 'под', 'над', 'между', 'перед',
    'около', 'вокруг', 'возле', 'после', 'вследствие', 'благодаря',
    'вопреки', 'наперекор', 'исключая', 'включая', 'ради', 'кроме',
    'подобно', 'согласно', 'посреди', 'среди', 'внутри', 'вне',
    'позади', 'напротив', 'против', 'из-за', 'из-под', 'мимо',
    'сквозь', 'близ', 'у', 'воз', 'к', 'поперек', 'напоперек'
}
STOP_WORDS = CONJUNCTIONS | PREPOSITIONS


segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_parser = None


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


def is_valid_token(token):
    token_lower = token.lower()

    if re.match(r'^\d+$', token_lower):
        return False

    if re.search(r'\d.*[а-яА-ЯёЁ]|[а-яА-ЯёЁ].*\d', token_lower):
        return False

    if re.match(r'^[^а-яА-ЯёЁ]+$', token_lower):
        return False

    if not re.search(r'[а-яА-ЯёЁ]', token_lower):
        return False

    if len(re.findall(r'[а-яА-ЯёЁ]', token_lower)) <= 1:
        return False

    if re.search(r'[<>&⟨⟩{}\[\]\\/|]', token_lower):
        return False

    return True


def split_hyphenated(token):
    parts = re.split(r'[-—]', token)
    return [p.strip() for p in parts if p.strip()]

TERM_OUTPUT_DIR = OUTPUT_DIR / 'terms'
LEMMA_OUTPUT_DIR = OUTPUT_DIR / 'lemmas'


def load_term_lists():
    term_lists = {}
    for doc_dir in sorted(INPUT_DIR.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 0):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        tokens_path = doc_dir / 'tokens.txt'
        if tokens_path.exists():
            with open(tokens_path, 'r', encoding='utf-8') as f:
                terms = set(line.strip() for line in f if line.strip())
            term_lists[doc_id] = terms
    return term_lists


def load_lemma_lists():
    lemma_lists = {}
    for doc_dir in sorted(INPUT_DIR.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 0):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        lemmas_path = doc_dir / 'lemmas.txt'
        if lemmas_path.exists():
            lemmas = {}
            with open(lemmas_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        lemma = parts[0]
                        lemmas[lemma] = parts[1:]
            lemma_lists[doc_id] = lemmas
    return lemma_lists


def count_term_frequencies(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    term_counts = defaultdict(int)
    total_tokens = 0

    for token in doc.tokens:
        original = token.text.strip('.,;:!?()[]""\'"…«»—')
        if not original:
            continue

        if not is_valid_token(original):
            continue

        original_lower = original.lower()

        if original_lower in STOP_WORDS:
            continue

        parts = split_hyphenated(original_lower)

        for part in parts:
            if not part:
                continue

            if not is_valid_token(part):
                continue

            if part in STOP_WORDS:
                continue

            term_counts[part] += 1
            total_tokens += 1

    return term_counts, total_tokens


def count_lemma_frequencies(text):
    global morph_parser
    if morph_parser is None:
        morph_parser = pymorphy2.MorphAnalyzer()

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    lemma_counts = defaultdict(int)
    total_tokens = 0

    for token in doc.tokens:
        original = token.text.strip('.,;:!?()[]""\'"…«»—')
        if not original:
            continue

        if not is_valid_token(original):
            continue

        original_lower = original.lower()

        if original_lower in STOP_WORDS:
            continue

        parts = split_hyphenated(original_lower)

        for part in parts:
            if not part:
                continue

            if not is_valid_token(part):
                continue

            if part in STOP_WORDS:
                continue

            parsed = morph_parser.parse(part)
            lemma = parsed[0].normal_form if parsed else part

            lemma_counts[lemma] += 1
            total_tokens += 1

    return lemma_counts, total_tokens


def main():
    TERM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LEMMA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading term and lemma lists from task2/output...')
    term_lists = load_term_lists()
    lemma_lists = load_lemma_lists()

    all_doc_ids = set(term_lists.keys()) | set(lemma_lists.keys())
    all_doc_ids = sorted(all_doc_ids, key=lambda x: int(x) if x.isdigit() else 0)

    N = len(all_doc_ids)

    print(f'\nProcessing {N} documents...')

    term_doc_freq = defaultdict(int)
    lemma_doc_freq = defaultdict(int)

    term_stats = {}
    lemma_stats = {}

    html_files = sorted(PAGES_DIR.glob('*.html'), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)

    for filepath in html_files:
        doc_id = filepath.stem
        if doc_id not in all_doc_ids:
            continue

        print(f'Processing document {doc_id}...', end='\r')

        text = extract_text_from_html(filepath)

        if text:
            term_counts, total_tokens = count_term_frequencies(text)
            lemma_counts, _ = count_lemma_frequencies(text)

            term_stats[doc_id] = {'counts': term_counts, 'total': total_tokens}
            lemma_stats[doc_id] = {'counts': lemma_counts, 'total': total_tokens}

            for term in term_counts:
                term_doc_freq[term] += 1

            for lemma in lemma_counts:
                lemma_doc_freq[lemma] += 1

    print('\nComputing IDF and TF-IDF...')

    term_idf = {}
    lemma_idf = {}

    for term, df in term_doc_freq.items():
        term_idf[term] = math.log10(N / df)

    for lemma, df in lemma_doc_freq.items():
        lemma_idf[lemma] = math.log10(N / df)

    print('Writing output files...')

    for doc_id in all_doc_ids:
        terms_path = TERM_OUTPUT_DIR / f'{doc_id}.txt'
        lemmas_path = LEMMA_OUTPUT_DIR / f'{doc_id}.txt'

        if doc_id in term_lists:
            terms = term_lists[doc_id]
            stats = term_stats.get(doc_id, {})
            term_counts = stats.get('counts', {})
            total_tokens = stats.get('total', 1)

            with open(terms_path, 'w', encoding='utf-8') as f:
                for term in sorted(terms):
                    tf = term_counts.get(term, 0) / total_tokens
                    idf = term_idf.get(term, 0)
                    tf_idf = tf * idf
                    f.write(f'{term} {idf:.6f} {tf_idf:.6f}\n')

        if doc_id in lemma_lists:
            lemmas = lemma_lists[doc_id]
            stats = lemma_stats.get(doc_id, {})
            lemma_counts = stats.get('counts', {})
            total_tokens = stats.get('total', 1)

            with open(lemmas_path, 'w', encoding='utf-8') as f:
                for lemma in sorted(lemmas.keys()):
                    tf = lemma_counts.get(lemma, 0) / total_tokens
                    idf = lemma_idf.get(lemma, 0)
                    tf_idf = tf * idf
                    f.write(f'{lemma} {idf:.6f} {tf_idf:.6f}\n')

    print(f'\nDone! Output written to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
