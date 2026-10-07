from __future__ import annotations

from collections.abc import Callable, Sequence

from Lexer import Token, TokenKind
from ast_nodes import (
    Assignment,
    BinaryExpr,
    BinaryOperator,
    Block,
    BoolLiteral,
    CallExpr,
    CallStmt,
    Expr,
    FunctionDecl,
    IdentifierExpr,
    IfStmt,
    IntLiteral,
    Node,
    Parameter,
    PrintItem,
    PrintStmt,
    Program,
    ReturnStmt,
    SourceSpan,
    Stmt,
    StringLiteral,
    TypeName,
    UnaryExpr,
    UnaryOperator,
    VarDecl,
    WhileStmt,
)


TYPE_START = {TokenKind.KW_INT, TokenKind.KW_BOOL, TokenKind.KW_VOID}
EXPRESSION_START = {
    TokenKind.IDENTIFIER,
    TokenKind.INT_LITERAL,
    TokenKind.KW_FALSE,
    TokenKind.KW_TRUE,
    TokenKind.LEFT_PAREN,
    TokenKind.LOGICAL_NOT,
    TokenKind.MINUS,
}
STATEMENT_START = TYPE_START | {
    TokenKind.IDENTIFIER,
    TokenKind.KW_IF,
    TokenKind.KW_WHILE,
    TokenKind.KW_RETURN,
    TokenKind.KW_PRINT,
    TokenKind.LEFT_BRACE,
}


TYPE_BY_TOKEN = {
    TokenKind.KW_INT: TypeName.INT,
    TokenKind.KW_BOOL: TypeName.BOOL,
    TokenKind.KW_VOID: TypeName.VOID,
}

UNARY_OPERATOR_BY_TOKEN = {
    TokenKind.LOGICAL_NOT: UnaryOperator.NOT,
    TokenKind.MINUS: UnaryOperator.NEGATE,
}

BINARY_OPERATOR_BY_TOKEN = {
    TokenKind.LOGICAL_OR: BinaryOperator.LOGICAL_OR,
    TokenKind.LOGICAL_AND: BinaryOperator.LOGICAL_AND,
    TokenKind.EQUAL_EQUAL: BinaryOperator.EQUAL,
    TokenKind.NOT_EQUAL: BinaryOperator.NOT_EQUAL,
    TokenKind.LESS: BinaryOperator.LESS,
    TokenKind.LESS_EQUAL: BinaryOperator.LESS_EQUAL,
    TokenKind.GREATER: BinaryOperator.GREATER,
    TokenKind.GREATER_EQUAL: BinaryOperator.GREATER_EQUAL,
    TokenKind.PLUS: BinaryOperator.ADD,
    TokenKind.MINUS: BinaryOperator.SUBTRACT,
    TokenKind.STAR: BinaryOperator.MULTIPLY,
    TokenKind.SLASH: BinaryOperator.DIVIDE,
    TokenKind.PERCENT: BinaryOperator.REMAINDER,
}


class ParserError(Exception):
    def __init__(self, token: Token, expected: set[TokenKind]):
        self.token = token
        self.expected = frozenset(expected)
        super().__init__()

    @property
    def line(self) -> int:
        return self.token.line

    @property
    def column(self) -> int:
        return self.token.column

    def __str__(self) -> str:
        names = ", ".join(kind.name for kind in sorted(
            self.expected,
            key=lambda kind: kind.value,
        ))
        return (
            f"erro sintático em {self.line}:{self.column}: esperado {{{names}}}, "
            f"encontrado {self.token.kind.name} ({self.token.lexeme!r})"
        )


class Parser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = list(tokens)
        if not self.tokens:
            raise ValueError("a sequência de tokens deve terminar em EOF")
        if self.tokens[-1].kind is not TokenKind.EOF:
            raise ValueError("o último token deve ser EOF")
        if any(token.kind is TokenKind.EOF for token in self.tokens[:-1]):
            raise ValueError("EOF deve aparecer uma única vez, no final")
        self.current = 0

    def peek(self, offset: int = 0) -> Token:
        index = min(self.current + offset, len(self.tokens) - 1)
        return self.tokens[index]

    def check(self, kind: TokenKind) -> bool:
        return self.peek().kind is kind

    def advance(self) -> Token:
        token = self.peek()
        if self.current < len(self.tokens) - 1:
            self.current += 1
        return token

    def match(self, *kinds: TokenKind) -> Token | None:
        if self.peek().kind in kinds:
            return self.advance()
        return None

    def expect(self, kinds: TokenKind | set[TokenKind]) -> Token:
        expected = kinds if isinstance(kinds, set) else {kinds}
        token = self.peek()
        if token.kind not in expected:
            raise ParserError(token, set(expected))
        return self.advance()

    @staticmethod
    def _token_span(token: Token) -> SourceSpan:
        return SourceSpan(
            token.line,
            token.column,
            token.line,
            token.column + len(token.lexeme),
        )

    @staticmethod
    def _start(value: Token | Node) -> tuple[int, int]:
        if isinstance(value, Node):
            return value.span.start_line, value.span.start_column
        return value.line, value.column

    @staticmethod
    def _end(value: Token | Node) -> tuple[int, int]:
        if isinstance(value, Node):
            return value.span.end_line, value.span.end_column
        return value.line, value.column + len(value.lexeme)

    @classmethod
    def _span(cls, first: Token | Node, last: Token | Node) -> SourceSpan:
        start_line, start_column = cls._start(first)
        end_line, end_column = cls._end(last)
        return SourceSpan(start_line, start_column, end_line, end_column)

    def parse(self) -> Program:
        return self.parse_program()

    # program ::= function* EOF
    def parse_program(self) -> Program:
        start = self.peek()
        functions: list[FunctionDecl] = []
        while self.peek().kind in TYPE_START:
            functions.append(self.parse_function())
        eof = self.expect(TokenKind.EOF)
        return Program(functions, span=self._span(start, eof))

    # function ::= type IDENTIFIER ... block
    def parse_function(self) -> FunctionDecl:
        start = self.peek()
        return_type = self.parse_type()
        name = self.expect(TokenKind.IDENTIFIER)
        self.expect(TokenKind.LEFT_PAREN)
        parameters = (
            self.parse_parameter_list()
            if self.peek().kind in TYPE_START
            else []
        )
        self.expect(TokenKind.RIGHT_PAREN)
        body = self.parse_block()
        return FunctionDecl(
            return_type,
            name.lexeme,
            parameters,
            body,
            span=self._span(start, body),
        )

    # type ::= KW_INT | KW_BOOL | KW_VOID
    def parse_type(self) -> TypeName:
        token = self.expect(TYPE_START)
        return TYPE_BY_TOKEN[token.kind]

    # parameter_list ::= parameter (COMMA parameter)*
    def parse_parameter_list(self) -> list[Parameter]:
        parameters = [self.parse_parameter()]
        while self.match(TokenKind.COMMA):
            parameters.append(self.parse_parameter())
        return parameters

    # parameter ::= type IDENTIFIER
    def parse_parameter(self) -> Parameter:
        start = self.peek()
        type_name = self.parse_type()
        name = self.expect(TokenKind.IDENTIFIER)
        return Parameter(type_name, name.lexeme, span=self._span(start, name))

    # block ::= LEFT_BRACE statement* RIGHT_BRACE
    def parse_block(self) -> Block:
        left = self.expect(TokenKind.LEFT_BRACE)
        statements: list[Stmt] = []
        while self.peek().kind in STATEMENT_START:
            statements.append(self.parse_statement())
        right = self.expect(TokenKind.RIGHT_BRACE)
        return Block(statements, span=self._span(left, right))

    # statement ::= declaration | id_or_call_statement | if_statement
    #             | while_statement | return_statement | print_statement | block
    def parse_statement(self) -> Stmt:
        token = self.peek()
        if token.kind in TYPE_START:
            return self.parse_declaration()
        if token.kind is TokenKind.IDENTIFIER:
            return self.parse_id_or_call_statement()
        if token.kind is TokenKind.KW_IF:
            return self.parse_if_statement()
        if token.kind is TokenKind.KW_WHILE:
            return self.parse_while_statement()
        if token.kind is TokenKind.KW_RETURN:
            return self.parse_return_statement()
        if token.kind is TokenKind.KW_PRINT:
            return self.parse_print_statement()
        if token.kind is TokenKind.LEFT_BRACE:
            return self.parse_block()
        raise ParserError(token, STATEMENT_START)

    # id_or_call_statement ::= IDENTIFIER (ASSIGN expression
    #                        | LEFT_PAREN arguments RIGHT_PAREN) SEMICOLON
    def parse_id_or_call_statement(self) -> Stmt:
        name = self.expect(TokenKind.IDENTIFIER)
        token = self.expect({TokenKind.ASSIGN, TokenKind.LEFT_PAREN})
        if token.kind is TokenKind.ASSIGN:
            value = self.parse_expression()
            semicolon = self.expect(TokenKind.SEMICOLON)
            target = IdentifierExpr(name.lexeme, span=self._token_span(name))
            return Assignment(target, value, span=self._span(name, semicolon))
        arguments = self.parse_arguments()
        right = self.expect(TokenKind.RIGHT_PAREN)
        semicolon = self.expect(TokenKind.SEMICOLON)
        call = CallExpr(name.lexeme, arguments, span=self._span(name, right))
        return CallStmt(call, span=self._span(name, semicolon))

    # declaration ::= type IDENTIFIER (ASSIGN expression)? SEMICOLON
    def parse_declaration(self) -> Stmt:
        start = self.peek()
        type_name = self.parse_type()
        name = self.expect(TokenKind.IDENTIFIER)
        initializer = None
        if self.match(TokenKind.ASSIGN):
            initializer = self.parse_expression()
        semicolon = self.expect(TokenKind.SEMICOLON)
        return VarDecl(
            type_name, name.lexeme, initializer, span=self._span(start, semicolon)
        )

    # if_statement ::= KW_IF LEFT_PAREN expression RIGHT_PAREN block (KW_ELSE block)?
    def parse_if_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_IF)
        self.expect(TokenKind.LEFT_PAREN)
        condition = self.parse_expression()
        self.expect(TokenKind.RIGHT_PAREN)
        then_block = self.parse_block()
        else_block = None
        if self.match(TokenKind.KW_ELSE):
            else_block = self.parse_block()
        end = then_block if else_block is None else else_block
        return IfStmt(condition, then_block, else_block, span=self._span(start, end))

    # while_statement ::= KW_WHILE LEFT_PAREN expression RIGHT_PAREN block
    def parse_while_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_WHILE)
        self.expect(TokenKind.LEFT_PAREN)
        condition = self.parse_expression()
        self.expect(TokenKind.RIGHT_PAREN)
        body = self.parse_block()
        return WhileStmt(condition, body, span=self._span(start, body))

    # return_statement ::= KW_RETURN expression? SEMICOLON
    def parse_return_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_RETURN)
        value = None
        if self.peek().kind in EXPRESSION_START:
            value = self.parse_expression()
        semicolon = self.expect(TokenKind.SEMICOLON)
        return ReturnStmt(value, span=self._span(start, semicolon))

    # print_statement ::= KW_PRINT LEFT_PAREN print_item (COMMA print_item)*
    #                     RIGHT_PAREN SEMICOLON
    def parse_print_statement(self) -> Stmt:
        start = self.expect(TokenKind.KW_PRINT)
        self.expect(TokenKind.LEFT_PAREN)
        items = [self.parse_print_item()]
        while self.match(TokenKind.COMMA):
            items.append(self.parse_print_item())
        self.expect(TokenKind.RIGHT_PAREN)
        semicolon = self.expect(TokenKind.SEMICOLON)
        return PrintStmt(items, span=self._span(start, semicolon))

    # print_item ::= expression | string_literals
    def parse_print_item(self) -> PrintItem:
        if self.check(TokenKind.STRING_LITERAL):
            return self.parse_string_literals()
        if self.peek().kind in EXPRESSION_START:
            return self.parse_expression()
        raise ParserError(
            self.peek(), EXPRESSION_START | {TokenKind.STRING_LITERAL}
        )

    # string_literals ::= STRING_LITERAL+
    def parse_string_literals(self) -> StringLiteral:
        first = self.expect(TokenKind.STRING_LITERAL)
        last = first
        parts = [first.value]
        while self.check(TokenKind.STRING_LITERAL):
            last = self.advance()
            parts.append(last.value)
        return StringLiteral("".join(parts), span=self._span(first, last))

    def _parse_left_associative(
        self,
        parse_operand: Callable[[], Expr],
        operators: set[TokenKind],
    ) -> Expr:
        left = parse_operand()
        while self.peek().kind in operators:
            operator = self.advance()
            right = parse_operand()
            left = BinaryExpr(
                BINARY_OPERATOR_BY_TOKEN[operator.kind],
                left,
                right,
                span=self._span(left, right),
            )
        return left

    # expression ::= logical_or
    def parse_expression(self) -> Expr:
        return self.parse_logical_or()

    # logical_or ::= logical_and (LOGICAL_OR logical_and)*
    def parse_logical_or(self) -> Expr:
        return self._parse_left_associative(
            self.parse_logical_and, {TokenKind.LOGICAL_OR}
        )

    # logical_and ::= equality (LOGICAL_AND equality)*
    def parse_logical_and(self) -> Expr:
        return self._parse_left_associative(
            self.parse_equality, {TokenKind.LOGICAL_AND}
        )

    # equality ::= relational ((EQUAL_EQUAL | NOT_EQUAL) relational)*
    def parse_equality(self) -> Expr:
        return self._parse_left_associative(
            self.parse_relational, {TokenKind.EQUAL_EQUAL, TokenKind.NOT_EQUAL}
        )

    # relational ::= additive ((LESS | LESS_EQUAL | GREATER | GREATER_EQUAL) additive)*
    def parse_relational(self) -> Expr:
        return self._parse_left_associative(
            self.parse_additive,
            {
                TokenKind.LESS,
                TokenKind.LESS_EQUAL,
                TokenKind.GREATER,
                TokenKind.GREATER_EQUAL,
            },
        )

    # additive ::= multiplicative ((PLUS | MINUS) multiplicative)*
    def parse_additive(self) -> Expr:
        return self._parse_left_associative(
            self.parse_multiplicative, {TokenKind.PLUS, TokenKind.MINUS}
        )

    # multiplicative ::= unary ((STAR | SLASH | PERCENT) unary)*
    def parse_multiplicative(self) -> Expr:
        return self._parse_left_associative(
            self.parse_unary, {TokenKind.STAR, TokenKind.SLASH, TokenKind.PERCENT}
        )

    # unary ::= (LOGICAL_NOT | MINUS) unary | primary
    def parse_unary(self) -> Expr:
        if self.peek().kind in UNARY_OPERATOR_BY_TOKEN:
            operator = self.advance()
            operand = self.parse_unary()
            return UnaryExpr(
                UNARY_OPERATOR_BY_TOKEN[operator.kind],
                operand,
                span=self._span(operator, operand),
            )
        return self.parse_primary()

    # primary ::= LEFT_PAREN expression RIGHT_PAREN
    #           | IDENTIFIER (LEFT_PAREN arguments RIGHT_PAREN)?
    #           | INT_LITERAL | KW_TRUE | KW_FALSE
    def parse_primary(self) -> Expr:
        token = self.peek()
        if token.kind is TokenKind.LEFT_PAREN:
            self.advance()
            inner = self.parse_expression()
            right = self.expect(TokenKind.RIGHT_PAREN)
            inner.span = self._span(token, right)
            return inner
        if token.kind is TokenKind.IDENTIFIER:
            self.advance()
            if self.match(TokenKind.LEFT_PAREN):
                arguments = self.parse_arguments()
                right = self.expect(TokenKind.RIGHT_PAREN)
                return CallExpr(token.lexeme, arguments, span=self._span(token, right))
            return IdentifierExpr(token.lexeme, span=self._token_span(token))
        if token.kind is TokenKind.INT_LITERAL:
            self.advance()
            return IntLiteral(token.value, span=self._token_span(token))
        if token.kind in {TokenKind.KW_TRUE, TokenKind.KW_FALSE}:
            self.advance()
            return BoolLiteral(
                token.kind is TokenKind.KW_TRUE, span=self._token_span(token)
            )
        raise ParserError(token, EXPRESSION_START)

    # arguments ::= (expression (COMMA expression)*)?
    def parse_arguments(self) -> list[Expr]:
        arguments: list[Expr] = []
        if self.peek().kind in EXPRESSION_START:
            arguments.append(self.parse_expression())
            while self.match(TokenKind.COMMA):
                arguments.append(self.parse_expression())
        return arguments

