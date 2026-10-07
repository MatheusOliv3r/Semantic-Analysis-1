from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterator


class TokenKind(enum.Enum):
    """Classe já implementada: nomes e números não devem ser alterados."""

    EOF = -1

    IDENTIFIER = 1
    INT_LITERAL = 2
    STRING_LITERAL = 3

    KW_INT = 10
    KW_BOOL = 11
    KW_VOID = 12
    KW_TRUE = 13
    KW_FALSE = 14
    KW_IF = 15
    KW_ELSE = 16
    KW_WHILE = 17
    KW_RETURN = 18
    KW_PRINT = 19

    PLUS = 20
    MINUS = 21
    STAR = 22
    SLASH = 23
    PERCENT = 24
    LESS = 25
    LESS_EQUAL = 26
    GREATER = 27
    GREATER_EQUAL = 28
    EQUAL_EQUAL = 29
    NOT_EQUAL = 30
    LOGICAL_AND = 31
    LOGICAL_OR = 32
    LOGICAL_NOT = 33
    ASSIGN = 34

    LEFT_PAREN = 40
    RIGHT_PAREN = 41
    LEFT_BRACE = 42
    RIGHT_BRACE = 43
    COMMA = 44
    SEMICOLON = 45


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    lexeme: str
    value: int | str | bool | None
    line: int
    column: int

    def __str__(self) -> str:
        return (
            f"<{self.kind.value}, {self.kind.name}, {self.lexeme!r}, "
            f"{self.value!r}, {self.line}, {self.column}>"
        )


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return f"erro léxico em {self.line}:{self.column}: {self.message}"


class Lexer:
    """Converte texto-fonte MicroC em uma sequência de tokens."""

    _KEYWORDS = {
        "int": TokenKind.KW_INT,
        "bool": TokenKind.KW_BOOL,
        "void": TokenKind.KW_VOID,
        "true": TokenKind.KW_TRUE,
        "false": TokenKind.KW_FALSE,
        "if": TokenKind.KW_IF,
        "else": TokenKind.KW_ELSE,
        "while": TokenKind.KW_WHILE,
        "return": TokenKind.KW_RETURN,
        "print": TokenKind.KW_PRINT,
    }

    _SINGLE_CHAR_TOKENS = {
        "+": TokenKind.PLUS,
        "-": TokenKind.MINUS,
        "*": TokenKind.STAR,
        "/": TokenKind.SLASH,
        "%": TokenKind.PERCENT,
        "<": TokenKind.LESS,
        ">": TokenKind.GREATER,
        "!": TokenKind.LOGICAL_NOT,
        "=": TokenKind.ASSIGN,
        "(": TokenKind.LEFT_PAREN,
        ")": TokenKind.RIGHT_PAREN,
        "{": TokenKind.LEFT_BRACE,
        "}": TokenKind.RIGHT_BRACE,
        ",": TokenKind.COMMA,
        ";": TokenKind.SEMICOLON,
    }

    _DOUBLE_CHAR_TOKENS = {
        "<=": TokenKind.LESS_EQUAL,
        ">=": TokenKind.GREATER_EQUAL,
        "==": TokenKind.EQUAL_EQUAL,
        "!=": TokenKind.NOT_EQUAL,
        "&&": TokenKind.LOGICAL_AND,
        "||": TokenKind.LOGICAL_OR,
    }

    _ESCAPES = {
        "n": "\n",
        "t": "\t",
        '"': '"',
        "\\": "\\",
    }

    def __init__(self, source: str):
        self.source = source
        self._index = 0
        self._line = 1
        self._column = 1

    def _at_end(self) -> bool:
        return self._index >= len(self.source)

    def _peek(self, distance: int = 0) -> str:
        index = self._index + distance
        if index >= len(self.source):
            return ""
        return self.source[index]

    def _advance(self) -> str:
        character = self.source[self._index]
        self._index += 1
        if character == "\n":
            self._line += 1
            self._column = 1
        else:
            self._column += 1
        return character

    @staticmethod
    def _is_identifier_start(character: str) -> bool:
        return (
            "a" <= character <= "z"
            or "A" <= character <= "Z"
            or character == "_"
        )

    @classmethod
    def _is_identifier_part(cls, character: str) -> bool:
        return cls._is_identifier_start(character) or "0" <= character <= "9"

    def _ensure_ascii(self) -> None:
        character = self._peek()
        if character and ord(character) > 127:
            raise LexerError("caractere não ASCII", self._line, self._column)

    def _skip_ignored(self) -> None:
        while not self._at_end():
            self._ensure_ascii()
            character = self._peek()

            if character in {" ", "\t", "\n"}:
                self._advance()
                continue

            if character == "/" and self._peek(1) == "/":
                self._advance()
                self._advance()
                while not self._at_end() and self._peek() != "\n":
                    self._ensure_ascii()
                    self._advance()
                continue

            if character == "/" and self._peek(1) == "*":
                start_line = self._line
                start_column = self._column
                self._advance()
                self._advance()
                while not self._at_end():
                    self._ensure_ascii()
                    if self._peek() == "*" and self._peek(1) == "/":
                        self._advance()
                        self._advance()
                        break
                    self._advance()
                else:
                    raise LexerError(
                        "comentário de bloco não terminado",
                        start_line,
                        start_column,
                    )
                continue

            return

    def _scan_identifier(self, line: int, column: int) -> Token:
        start = self._index
        while self._is_identifier_part(self._peek()):
            self._advance()

        lexeme = self.source[start:self._index]
        kind = self._KEYWORDS.get(lexeme, TokenKind.IDENTIFIER)
        if kind is TokenKind.KW_TRUE:
            value: int | str | bool | None = True
        elif kind is TokenKind.KW_FALSE:
            value = False
        elif kind is TokenKind.IDENTIFIER:
            value = lexeme
        else:
            value = None
        return Token(kind, lexeme, value, line, column)

    def _scan_integer(self, line: int, column: int) -> Token:
        start = self._index
        while "0" <= self._peek() <= "9":
            self._advance()

        lexeme = self.source[start:self._index]
        return Token(TokenKind.INT_LITERAL, lexeme, int(lexeme), line, column)

    def _scan_string(self, line: int, column: int) -> Token:
        start = self._index
        decoded: list[str] = []
        self._advance()

        while not self._at_end():
            self._ensure_ascii()
            character = self._peek()

            if character == '"':
                self._advance()
                lexeme = self.source[start:self._index]
                return Token(
                    TokenKind.STRING_LITERAL,
                    lexeme,
                    "".join(decoded),
                    line,
                    column,
                )

            if character == "\n":
                raise LexerError("quebra de linha em string", self._line, self._column)

            if character == "\\":
                escape_line = self._line
                escape_column = self._column
                self._advance()
                if self._at_end():
                    raise LexerError("string não terminada", line, column)
                self._ensure_ascii()
                escaped = self._peek()
                if escaped == "\n":
                    raise LexerError(
                        "quebra de linha em string",
                        self._line,
                        self._column,
                    )
                if escaped not in self._ESCAPES:
                    raise LexerError("escape inválido", escape_line, escape_column)
                self._advance()
                decoded.append(self._ESCAPES[escaped])
                continue

            decoded.append(self._advance())

        raise LexerError("string não terminada", line, column)

    def tokens(self) -> Iterator[Token]:
        """Produza todos os tokens significativos e um único EOF ao final."""
        self._index = 0
        self._line = 1
        self._column = 1

        while True:
            self._skip_ignored()
            if self._at_end():
                yield Token(TokenKind.EOF, "", None, self._line, self._column)
                return

            self._ensure_ascii()
            line = self._line
            column = self._column
            character = self._peek()

            if self._is_identifier_start(character):
                yield self._scan_identifier(line, column)
                continue

            if "0" <= character <= "9":
                yield self._scan_integer(line, column)
                continue

            if character == '"':
                yield self._scan_string(line, column)
                continue

            pair = character + self._peek(1)
            if pair in self._DOUBLE_CHAR_TOKENS:
                self._advance()
                self._advance()
                yield Token(self._DOUBLE_CHAR_TOKENS[pair], pair, None, line, column)
                continue

            if character in {"&", "|"}:
                raise LexerError(f"operador incompleto {character!r}", line, column)

            kind = self._SINGLE_CHAR_TOKENS.get(character)
            if kind is not None:
                self._advance()
                yield Token(kind, character, None, line, column)
                continue

            raise LexerError(f"caractere inválido {character!r}", line, column)

    def scan(self) -> list[Token]:
        return list(self.tokens())
