from flask import Flask, request, jsonify, render_template
from pathlib import Path
from typing import Set
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from task5.vector_search import VectorSearchEngine
from task2.tokenize_lemmatize import extract_text_from_html, STOP_WORDS
from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab

app = Flask(__name__)

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

PAGES_DIR = Path(__file__).parent.parent.resolve() / 'task1' / 'pages'

try:
    engine = VectorSearchEngine()
except FileNotFoundError as e:
    print(f'Ошибка: {e}')
    print('Сначала создайте индекс в task5/')
    sys.exit(1)


def load_document_text(doc_id: str) -> str:
    html_file = PAGES_DIR / f'{doc_id}.html'
    if html_file.exists():
        return extract_text_from_html(html_file)
    return ''


def lemmatize_query(query: str) -> Set[str]:
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

        token.lemmatize(morph_vocab)
        lemma = token.lemma.lower() if token.lemma else original_lower

        if lemma in engine.vocabulary:
            lemmas.add(lemma)

    return lemmas


def create_snippet(text: str, query_lemmas: Set[str], max_length: int = 300) -> str:
    sentences = re.split(r'[.!?]+', text)

    best_snippets = []
    seen_lemmas = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_lemmas = set()
        for token in sentence.lower().split():
            token_clean = re.sub(r'[^\wа-яА-ЯёЁ]', '', token)
            if token_clean:
                doc = Doc(token_clean)
                doc.segment(segmenter)
                doc.tag_morph(morph_tagger)
                if doc.tokens:
                    doc.tokens[0].lemmatize(morph_vocab)
                    lemma = doc.tokens[0].lemma.lower() if doc.tokens[0].lemma else token_clean
                    if lemma in query_lemmas:
                        sentence_lemmas.add(lemma)

        if sentence_lemmas and not sentence_lemmas.issubset(seen_lemmas):
            best_snippets.append((sentence, len(sentence_lemmas)))
            seen_lemmas.update(sentence_lemmas)

    best_snippets.sort(key=lambda x: -x[1])

    if not best_snippets:
        if len(text) > max_length:
            return text[:max_length] + '...'
        return text

    snippets = [s[0] for s in best_snippets[:2]]
    combined = '... '.join(snippets)

    if len(combined) > max_length:
        combined = combined[:max_length] + '...'

    return combined


def highlight_terms(snippet: str, query: str) -> str:
    doc = Doc(query.lower())
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    terms = set()
    for token in doc.tokens:
        original = token.text.strip('.,;:!?()[]""\'"…«»—')
        if original and original.lower() not in STOP_WORDS:
            token.lemmatize(morph_vocab)
            lemma = token.lemma.lower() if token.lemma else original.lower()
            terms.add(lemma)
            terms.add(original.lower())

    for term in sorted(terms, key=len, reverse=True):
        snippet = re.sub(
            f'({re.escape(term)})',
            r'<mark>\1</mark>',
            snippet,
            flags=re.IGNORECASE
        )

    return snippet


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/search', methods=['GET'])
def api_search():
    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 10))

    if not query:
        return jsonify({'error': 'Пустой запрос'}), 400

    results = engine.search(query, top_k=limit)

    enriched_results = []
    query_lemmas = lemmatize_query(query)

    for doc_id, score in results:
        doc_text = load_document_text(doc_id)

        if doc_text:
            snippet = create_snippet(doc_text, query_lemmas)
            snippet_highlighted = highlight_terms(snippet, query)
        else:
            snippet_highlighted = 'Текст документа недоступен'

        enriched_results.append({
            'doc_id': doc_id,
            'score': round(score, 4),
            'snippet': snippet_highlighted
        })

    return jsonify({
        'query': query,
        'total': len(enriched_results),
        'results': enriched_results
    })


@app.route('/api/document/<doc_id>', methods=['GET'])
def api_document(doc_id):
    doc_text = load_document_text(doc_id)

    if not doc_text:
        return jsonify({'error': 'Документ не найден'}), 404

    return jsonify({
        'doc_id': doc_id,
        'text': doc_text
    })


if __name__ == '__main__':
    print('\n=== Векторный поиск - Web интерфейс ===\n')
    print('Запуск сервера на http://localhost:5001\n')
    app.run(debug=True, host='0.0.0.0', port=5001)
