import json
from pathlib import Path
from collections import defaultdict
import math


INPUT_DIR = Path(__file__).parent.parent / 'task4' / 'output' / 'lemmas'
OUTPUT_DIR = Path(__file__).parent
VECTOR_INDEX_FILE = OUTPUT_DIR / 'vectors.json'


def load_lemma_tfidf():
    doc_vectors = {}
    all_lemmas = set()
    lemma_idf = {}

    lemmas_files = sorted(INPUT_DIR.glob('*.txt'), key=lambda x: int(x.stem) if x.stem.isdigit() else 999999)

    for filepath in lemmas_files:
        doc_id = filepath.stem
        print(f'  Обработка файла {doc_id}...', end='\r')

        doc_vector = {}

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    lemma = parts[0]
                    idf = float(parts[1])
                    tfidf = float(parts[2])

                    doc_vector[lemma] = tfidf
                    all_lemmas.add(lemma)
                    lemma_idf[lemma] = idf

        if doc_vector:
            doc_vectors[doc_id] = doc_vector

    return doc_vectors, all_lemmas, lemma_idf


def build_vocabulary(all_lemmas):
    vocabulary = {}
    for idx, lemma in enumerate(sorted(all_lemmas)):
        vocabulary[lemma] = idx

    return vocabulary


def build_sparse_vectors(doc_vectors, vocabulary, lemma_idf):
    sparse_vectors = {}
    doc_norms = {}

    print('Построение векторов и вычисление норм...')

    for doc_id, lemma_tfidf in doc_vectors.items():
        print(f'  Документ {doc_id}...', end='\r')

        sparse_vector = {}
        squared_sum = 0.0

        for lemma, tfidf in lemma_tfidf.items():
            if lemma in vocabulary:
                idx = vocabulary[lemma]
                sparse_vector[idx] = tfidf
                squared_sum += tfidf * tfidf

        sparse_vectors[doc_id] = sparse_vector
        doc_norms[doc_id] = math.sqrt(squared_sum)

    print(f'\nПостроено {len(sparse_vectors)} векторов')

    return sparse_vectors, doc_norms


def save_vector_index(vocabulary, doc_vectors, doc_norms, lemma_idf, output_path):
    index_data = {
        'vocabulary': vocabulary,
        'doc_vectors': {str(k): v for k, v in doc_vectors.items()},
        'doc_norms': {str(k): v for k, v in doc_norms.items()},
        'idf': lemma_idf,
        'num_documents': len(doc_vectors),
        'vocabulary_size': len(vocabulary)
    }

    print(f'Сохранение индекса в {output_path}...')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    doc_vectors, all_lemmas, lemma_idf = load_lemma_tfidf()
    vocabulary = build_vocabulary(all_lemmas)
    sparse_vectors, doc_norms = build_sparse_vectors(doc_vectors, vocabulary, lemma_idf)

    save_vector_index(vocabulary, sparse_vectors, doc_norms, lemma_idf, VECTOR_INDEX_FILE)

    print(f'  Документов: {len(sparse_vectors)}')
    print(f'  Размерность вектора: {len(vocabulary)}')


if __name__ == '__main__':
    main()
