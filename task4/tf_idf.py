import sys
import math
from pathlib import Path
from collections import defaultdict
from natasha import Doc

sys.path.insert(0, str(Path(__file__).parent.parent / 'task2'))


from tokenize_lemmatize import (
    PAGES_DIR, extract_text_from_html, is_valid_token,
    split_hyphenated, STOP_WORDS, segmenter, morph_tagger
)


INPUT_DIR = Path(__file__).parent.parent / 'task2' / 'output'
OUTPUT_DIR = Path(__file__).parent / 'output'

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
    token_to_lemma = {}

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
                        forms = parts[1:]
                        lemmas[lemma] = forms
                        for form in forms:
                            token_to_lemma[form] = lemma
            lemma_lists[doc_id] = lemmas
    return lemma_lists, token_to_lemma


def count_frequencies(text, doc_terms_set, token_to_lemma):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    term_counts = defaultdict(int)
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

            if part in doc_terms_set:
                term_counts[part] += 1
                total_tokens += 1

            if part in token_to_lemma:
                lemma = token_to_lemma[part]
                lemma_counts[lemma] += 1

    return term_counts, lemma_counts, total_tokens


def main():
    TERM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LEMMA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading term and lemma lists from task2/output...')
    term_lists = load_term_lists()
    lemma_lists, token_to_lemma = load_lemma_lists()

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
            doc_terms_set = term_lists.get(doc_id, set())
            term_counts, lemma_counts, total_tokens = count_frequencies(text, doc_terms_set, token_to_lemma)

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
