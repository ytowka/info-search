import re
from dataclasses import dataclass
from typing import List


@dataclass
class Token:
    type: str  # TERM, AND, OR, NOT, LPAREN, RPAREN, EOF
    value: str
    position: int


class Lexer:
    TOKEN_PATTERNS = [
        (r'\bAND\b', 'AND'),
        (r'\bOR\b', 'OR'),
        (r'\bNOT\b', 'NOT'),
        (r'\(', 'LPAREN'),
        (r'\)', 'RPAREN'),
        (r'"([^"]*)"', 'TERM'),  # Фраза в кавычках
        (r'[^\s()]+', 'TERM'),   # Термин (любая последовательность, кроме пробелов и скобок)
    ]
    
    def __init__(self, text: str):
        self.text = text
        self.position = 0
        self.tokens: List[Token] = []
        self.current_char = self.text[0] if text else None
    
    def advance(self):
        self.position += 1
        if self.position < len(self.text):
            self.current_char = self.text[self.position]
        else:
            self.current_char = None
    
    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def get_token(self) -> Token:
        self.skip_whitespace()
        
        if self.position >= len(self.text):
            return Token('EOF', '', self.position)
        
        token_start = self.position
        
        for pattern, token_type in self.TOKEN_PATTERNS:
            regex = re.compile(pattern, re.IGNORECASE)
            match = regex.match(self.text[self.position:])
            
            if match:
                matched_text = match.group()
                self.position += len(matched_text)
                self.current_char = self.text[self.position] if self.position < len(self.text) else None
                
                if token_type == 'TERM':
                    value = matched_text.strip('"')
                else:
                    value = matched_text.upper()
                
                return Token(token_type, value, token_start)
        
        raise SyntaxError(f"Неизвестный токен на позиции {self.position}: '{self.text[self.position]}'")
    
    def tokenize(self) -> List[Token]:
        self.tokens = []
        self.position = 0
        self.current_char = self.text[0] if self.text else None
        
        while True:
            token = self.get_token()
            self.tokens.append(token)
            if token.type == 'EOF':
                break
        
        return self.tokens


@dataclass
class ASTNode:
    pass


@dataclass
class TermNode(ASTNode):
    term: str


@dataclass
class AndNode(ASTNode):
    left: ASTNode
    right: ASTNode


@dataclass
class OrNode(ASTNode):
    left: ASTNode
    right: ASTNode


@dataclass
class NotNode(ASTNode):
    operand: ASTNode


class Parser:
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else Token('EOF', '', 0)
    
    def eat(self, token_type: str):
        if self.current_token.type == token_type:
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
            else:
                self.current_token = Token('EOF', '', self.pos)
        else:
            raise SyntaxError(
                f"Ожидается токен {token_type}, но получен {self.current_token.type} "
                f"на позиции {self.current_token.position}"
            )
    
    def parse(self) -> ASTNode:
        if len(self.tokens) == 0 or (len(self.tokens) == 1 and self.tokens[0].type == 'EOF'):
            raise SyntaxError("Пустой запрос")
        
        return self.parse_or()
    
    def parse_or(self) -> ASTNode:
        node = self.parse_and()
        
        while self.current_token.type == 'OR':
            self.eat('OR')
            right = self.parse_and()
            node = OrNode(node, right)
        
        return node
    
    def parse_and(self) -> ASTNode:
        node = self.parse_not()
        
        while self.current_token.type == 'AND':
            self.eat('AND')
            right = self.parse_not()
            node = AndNode(node, right)
        
        return node
    
    def parse_not(self) -> ASTNode:
        if self.current_token.type == 'NOT':
            self.eat('NOT')
            operand = self.parse_factor()
            return NotNode(operand)
        
        return self.parse_factor()
    
    def parse_factor(self) -> ASTNode:
        if self.current_token.type == 'LPAREN':
            self.eat('LPAREN')
            node = self.parse_or()
            self.eat('RPAREN')
            return node
        
        if self.current_token.type == 'TERM':
            term = self.current_token.value
            self.eat('TERM')
            return TermNode(term)
        
        if self.current_token.type == 'EOF':
            raise SyntaxError(f"Неожиданный конец выражения на позиции {self.current_token.position}")
        
        raise SyntaxError(
            f"Неожиданный токен {self.current_token.type} "
            f"на позиции {self.current_token.position}"
        )


class QueryParser:
    
    def __init__(self):
        self.lexer = None
        self.parser = None
    
    def parse(self, query: str) -> ASTNode:
        if not query or query.strip() == '':
            raise SyntaxError("Пустой запрос")
        
        self.lexer = Lexer(query)
        tokens = self.lexer.tokenize()
        
        self.parser = Parser(tokens)
        ast = self.parser.parse()
        
        return ast
    
    def get_tokens(self, query: str) -> List[Token]:
        lexer = Lexer(query)
        return lexer.tokenize()
    
    def ast_to_string(self, node: ASTNode, indent: int = 0) -> str:
        prefix = "  " * indent
        
        if isinstance(node, TermNode):
            return f"{prefix}Term({node.term})"
        elif isinstance(node, AndNode):
            return f"{prefix}AND(\n{self.ast_to_string(node.left, indent + 1)}\n{self.ast_to_string(node.right, indent + 1)}\n{prefix})"
        elif isinstance(node, OrNode):
            return f"{prefix}OR(\n{self.ast_to_string(node.left, indent + 1)}\n{self.ast_to_string(node.right, indent + 1)}\n{prefix})"
        elif isinstance(node, NotNode):
            return f"{prefix}NOT(\n{self.ast_to_string(node.operand, indent + 1)}\n{prefix})"
        else:
            return f"{prefix}Unknown({node})"


def main():
    parser = QueryParser()
    
    test_queries = [
        "Клеопатра AND Цезарь",
        "(Клеопатра AND Цезарь) OR (Антоний AND Цицерон) OR Помпей",
        "Помпей AND NOT Цезарь",
        "Клеопатра OR Цезарь OR Антоний",
        "NOT Клеопатра",
    ]
    
    for query in test_queries:
        print(f"\nЗапрос: {query}")
        try:
            ast = parser.parse(query)
            print("AST:")
            print(parser.ast_to_string(ast))
        except SyntaxError as e:
            print(f"Ошибка: {e}")


if __name__ == '__main__':
    main()
