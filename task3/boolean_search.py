import json
from pathlib import Path
from typing import Dict, Set, List, Optional

from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab

from query_parser import ASTNode, TermNode, AndNode, OrNode, NotNode


INDEX_FILE = Path(__file__).parent / 'inverted_index.json'

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()


class BooleanSearchEngine:
    
    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or INDEX_FILE
        self.index: Dict[str, Dict[str, List[int]]] = {}
        self.all_documents: Set[str] = set()
        self.lemmatization_cache: Dict[str, str] = {}
        
        self.load_index()
    
    def load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Файл индекса не найден: {self.index_path}")
        
        print(f"Загрузка индекса из {self.index_path}...")
        
        with open(self.index_path, 'r', encoding='utf-8') as f:
            self.index = json.load(f)
        
        print(f"Загружено {len(self.index)} лемм в индексе")
        
        self.all_documents = set()
        for lemma, docs in self.index.items():
            self.all_documents.update(docs.keys())
        
        print(f"Всего документов в индексе: {len(self.all_documents)}")
    
    def lemmatize_term(self, term: str) -> str:
        if term in self.lemmatization_cache:
            return self.lemmatization_cache[term]
        
        doc = Doc(term.lower())
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        
        if doc.tokens:
            doc.tokens[0].lemmatize(morph_vocab)
            lemma = doc.tokens[0].lemma.lower() if doc.tokens[0].lemma else term.lower()
        else:
            lemma = term.lower()
        
        self.lemmatization_cache[term] = lemma
        return lemma
    
    def get_documents_for_term(self, term: str) -> Set[str]:
        lemma = self.lemmatize_term(term)
        
        if lemma in self.index:
            return set(self.index[lemma].keys())
        
        return set()
    
    def execute_ast(self, ast_node: ASTNode) -> Set[str]:
        if isinstance(ast_node, TermNode):
            return self.get_documents_for_term(ast_node.term)
        
        elif isinstance(ast_node, AndNode):
            left_docs = self.execute_ast(ast_node.left)
            right_docs = self.execute_ast(ast_node.right)
            return left_docs & right_docs  # Пересечение
        
        elif isinstance(ast_node, OrNode):
            left_docs = self.execute_ast(ast_node.left)
            right_docs = self.execute_ast(ast_node.right)
            return left_docs | right_docs  # Объединение
        
        elif isinstance(ast_node, NotNode):
            operand_docs = self.execute_ast(ast_node.operand)
            return self.all_documents - operand_docs  # Разность
        
        else:
            raise ValueError(f"Неизвестный тип узла AST: {type(ast_node)}")
    
    def search(self, query_ast: ASTNode) -> List[str]:
        result_set = self.execute_ast(query_ast)
        
        return sorted(result_set, key=lambda x: int(x) if x.isdigit() else x)
    
    def get_positions(self, term: str, doc_id: str) -> List[int]:
        lemma = self.lemmatize_term(term)
        
        if lemma in self.index and doc_id in self.index[lemma]:
            return self.index[lemma][doc_id]
        
        return []
    
    def get_document_info(self, doc_id: str) -> Dict[str, any]:
        return {
            'doc_id': doc_id,
            'total_terms': sum(len(positions) for positions in self.index.values() if doc_id in positions),
        }
    
    def get_statistics(self) -> Dict[str, any]:
        total_lemmas = len(self.index)
        total_documents = len(self.all_documents)
        total_entries = sum(len(docs) for docs in self.index.values())
        avg_docs_per_lemma = total_entries / total_lemmas if total_lemmas > 0 else 0
        
        return {
            'total_lemmas': total_lemmas,
            'total_documents': total_documents,
            'total_entries': total_entries,
            'avg_docs_per_lemma': avg_docs_per_lemma,
        }


def main():
    try:
        engine = BooleanSearchEngine()
        
        print("\nСтатистика индекса:")
        stats = engine.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print("\nТестовый поиск:")
        test_queries = [
            "Клеопатра",
            "Цезарь",
            "Клеопатра AND Цезарь",
        ]
        
        from query_parser import QueryParser
        parser = QueryParser()
        
        for query in test_queries:
            print(f"\nЗапрос: {query}")
            try:
                ast = parser.parse(query)
                results = engine.search(ast)
                print(f"Результаты ({len(results)} документов): {results}")
            except Exception as e:
                print(f"Ошибка: {e}")
    
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("Сначала создайте индекс с помощью create_index.py")


if __name__ == '__main__':
    main()
