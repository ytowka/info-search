import sys
from pathlib import Path

from boolean_search import BooleanSearchEngine
from query_parser import QueryParser


class SearchInterface:
    
    def __init__(self: Path = None):
        self.parser = QueryParser()
        try:
            self.engine = BooleanSearchEngine()
        except FileNotFoundError as e:
            print(f"Ошибка: {e}")
            print("Сначала создайте индекс с помощью:")
            print("  python create_index.py")
            sys.exit(1)
    
    def search(self, query: str, show_ast: bool = False, show_positions: bool = False, 
               doc_id: str = None) -> dict:

        result = {
            'query': query,
            'success': False,
            'results': [],
            'ast': None,
            'positions': None,
            'error': None
        }
        
        try:
            ast = self.parser.parse(query)
            result['ast'] = self.parser.ast_to_string(ast)
            
            if show_ast:
                print(f"\nAST для запроса '{query}':")
                print(result['ast'])
                print()
            
            documents = self.engine.search(ast)
            result['results'] = documents
            result['success'] = True
            
            if show_positions and doc_id:
                if doc_id in documents:
                    terms = self._extract_terms_from_ast(ast)
                    positions = {}
                    for term in terms:
                        pos = self.engine.get_positions(term, doc_id)
                        if pos:
                            positions[term] = pos
                    
                    if positions:
                        result['positions'] = positions
            
        except SyntaxError as e:
            result['error'] = f"Синтаксическая ошибка: {e}"
        except Exception as e:
            result['error'] = f"Ошибка: {e}"
        
        return result
    
    def _extract_terms_from_ast(self, node) -> list:
        from query_parser import TermNode, AndNode, OrNode, NotNode
        
        if isinstance(node, TermNode):
            return [node.term]
        elif isinstance(node, (AndNode, OrNode)):
            return self._extract_terms_from_ast(node.left) + self._extract_terms_from_ast(node.right)
        elif isinstance(node, NotNode):
            return self._extract_terms_from_ast(node.operand)
        return []
    
    def format_results(self, result: dict, show_ast: bool = False, 
                       show_positions: bool = False, doc_id: str = None):
        
        if result['error']:
            print(f" {result['error']}")
            return
        
        print(f"Найдено документов: {len(result['results'])}")
        
        if result['results']:
            print(f"\nДокументы: {', '.join(result['results'])}")
            
            if show_positions and doc_id and result['positions']:
                print(f"\nПозиции в документе {doc_id}:")
                for term, positions in result['positions'].items():
                    print(f"  '{term}': {positions}")
        
        print()
    
    def interactive_mode(self):
        print("\nКоманды:")
        print("  ast - показать AST для следующего запроса")
        print("  help - показать справку")
        print()
        
        show_ast = False
        
        while True:
            try:
                if show_ast:
                    prompt = "Поиск [AST]: "
                else:
                    prompt = "Поиск: "
                
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'help':
                    print("  AND - пересечение (логическое И)")
                    print("  OR - объединение (логическое ИЛИ)")
                    print("  NOT - отрицание")
                    print("  () - скобки для группировки")
                    print("  ast - переключить режим показа AST")
                    print("  quit - выход")
                    continue
                
                if user_input.lower() == 'ast':
                    show_ast = not show_ast
                    mode = "включен" if show_ast else "выключен"
                    print(f"Режим показа AST {mode}")
                    continue
                
                result = self.search(user_input, show_ast=show_ast)
                self.format_results(result, show_ast=show_ast)

            except Exception as e:
                print(f"\n Ошибка: {e}\n")


def main():
    interface = SearchInterface()
    interface.interactive_mode()


if __name__ == '__main__':
    main()
