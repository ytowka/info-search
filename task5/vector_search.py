import json
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab

sys.path.insert(0, str(Path(__file__).parent.parent))
from task2.tokenize_lemmatize import STOP_WORDS


segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()


VECTOR_INDEX_FILE = Path(__file__).parent / 'vectors.json'


class VectorSearchEngine:

    def __init__(self, index_path: Path = None):
        self.index_path = index_path or VECTOR_INDEX_FILE
        self.vocabulary = {}
        self.doc_vectors = {}
        self.doc_norms = {}
        self.idf = {}
        self.num_documents = 0
        self.vocabulary_size = 0
        self.lemmatization_cache = {}

        self.load_index()

    def load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f'Файл индекса не найден: {self.index_path}')

        print(f'Загрузка индекса из {self.index_path}...')

        with open(self.index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        self.vocabulary = index_data['vocabulary']
        
        self.doc_vectors = {}
        for doc_id, vec in index_data['doc_vectors'].items():
            self.doc_vectors[doc_id] = {int(idx): tfidf for idx, tfidf in vec.items()}
        
        self.doc_norms = {k: v for k, v in index_data['doc_norms'].items()}
        self.idf = index_data['idf']
        self.num_documents = index_data['num_documents']
        self.vocabulary_size = index_data['vocabulary_size']

        print(f'Загружено {self.num_documents} документов')
        print(f'Размерность вектора: {self.vocabulary_size}')

    def lemmatize_query(self, query: str) -> Set[str]:
        doc = Doc(query.lower())
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)

        lemmas = set()

        for token in doc.tokens:
            original = token.text.strip('.,;:!?()[]""\'"…«»—')
            if not original:
                continue

            original_lower = original.lower()
            if original_lower in STOP_WORDS:
                continue

            if original_lower in self.lemmatization_cache:
                lemma = self.lemmatization_cache[original_lower]
            else:
                token.lemmatize(morph_vocab)
                lemma = token.lemma.lower() if token.lemma else original_lower
                self.lemmatization_cache[original_lower] = lemma

            if lemma in self.vocabulary:
                lemmas.add(lemma)

        return lemmas

    def build_query_vector(self, query_lemmas: Set[str]) -> Tuple[Dict[int, float], float]:
        query_vector = {}
        lemma_freq = defaultdict(int)

        for lemma in query_lemmas:
            lemma_freq[lemma] += 1

        total_terms = sum(lemma_freq.values())

        squared_sum = 0.0

        for lemma, freq in lemma_freq.items():
            if lemma in self.vocabulary:
                idx = self.vocabulary[lemma]
                tf = freq / total_terms
                idf = self.idf.get(lemma, 0.0)
                tfidf = tf * idf

                query_vector[idx] = tfidf
                squared_sum += tfidf * tfidf

        query_norm = math.sqrt(squared_sum)

        return query_vector, query_norm

    def cosine_similarity(self, query_vector: Dict[int, float], query_norm: float,
                         doc_vector: Dict[int, float], doc_norm: float) -> float:
        if query_norm == 0 or doc_norm == 0:
            return 0.0

        dot_product = 0.0

        for idx, query_tfidf in query_vector.items():
            if idx in doc_vector:
                dot_product += query_tfidf * doc_vector[idx]

        return dot_product / (query_norm * doc_norm)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_lemmas = self.lemmatize_query(query)

        if not query_lemmas:
            return []

        query_vector, query_norm = self.build_query_vector(query_lemmas)

        results = []

        for doc_id, doc_vector in self.doc_vectors.items():
            doc_norm = self.doc_norms[doc_id]
            score = self.cosine_similarity(query_vector, query_norm, doc_vector, doc_norm)

            if score > 0:
                results.append((doc_id, score))

        results.sort(key=lambda x: -x[1])

        return results[:top_k]


class SearchInterface:

    def __init__(self):
        self.top_k = 10

        try:
            self.engine = VectorSearchEngine()
        except FileNotFoundError as e:
            print(f'Ошибка: {e}')
            print('Сначала создайте индекс с помощью:')
            print('  python create_vector_index.py')
            exit(1)

    def search(self, query: str) -> dict:
        result = {
            'query': query,
            'success': False,
            'results': [],
            'error': None
        }

        try:
            results = self.engine.search(query, top_k=self.top_k)
            result['results'] = results
            result['success'] = True
        except Exception as e:
            result['error'] = f'Ошибка: {e}'

        return result

    def format_results(self, result: dict):
        if result['error']:
            print(f' {result["error"]}')
            return

        results = result['results']

        print(f'\nНайдено документов: {len(results)}')
        print(f'Топ-{min(self.top_k, len(results))} по релевантности:\n')

        for i, (doc_id, score) in enumerate(results, 1):
            print(f'{i}. Документ {doc_id} (score: {score:.4f})')

        print()

    def interactive_mode(self):
        print('\nВекторный поиск по документам')
        print('Команды:')
        print('  help   - показать справку')
        print('  top N  - изменить количество результатов')
        print('  quit   - выход')
        print()

        while True:
            try:
                user_input = input('Поиск: ').strip()

                if not user_input:
                    continue

                if user_input.lower() == 'help':
                    print('\nСправка:')
                    print('  Введите запрос на русском языке')
                    print('  Документы будут ранжированы по косинусному сходству')
                    print('  top N - изменить количество результатов (например, top 5)')
                    print('  quit - выход из программы\n')
                    continue

                if user_input.lower() == 'quit':
                    break

                if user_input.lower().startswith('top '):
                    try:
                        new_top_k = int(user_input.split()[1])
                        if new_top_k > 0:
                            self.top_k = new_top_k
                            print(f'Количество результатов изменено на {self.top_k}\n')
                        else:
                            print('Количество должно быть положительным числом\n')
                    except (IndexError, ValueError):
                        print('Неверный формат. Используйте: top N\n')
                    continue

                result = self.search(user_input)
                self.format_results(result)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f'\n Ошибка: {e}\n')


def main():
    interface = SearchInterface()
    interface.interactive_mode()


if __name__ == '__main__':
    main()
