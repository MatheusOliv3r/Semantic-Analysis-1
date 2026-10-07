from __future__ import annotations

from ast_nodes import (
    Assignment,
    BinaryExpr,
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
    PrintStmt,
    Program,
    ReturnStmt,
    Stmt,
    StringLiteral,
    TypeName,
    UnaryExpr,
    VarDecl,
    WhileStmt,
)
from semantic_errors import SemanticDiagnostic, SemanticError, SemanticErrorKind
from symbols import FunctionSymbol, Scope, Symbol, SymbolKind


def resolve_names(program: Program) -> None:
    """Construa escopos, símbolos e vínculos entre usos e declarações."""

    NameResolver().resolve(program)


def lookup(scope: Scope | None, name: str) -> Symbol | None:
    """Procura um nome do escopo mais interno até o mais externo."""

    while scope is not None:
        symbol = scope.symbols.get(name)
        if symbol is not None:
            return symbol
        scope = scope.parent
    return None


class NameResolver:
    def __init__(self) -> None:
        self.functions: dict[str, FunctionSymbol] = {}
        self.diagnostics: list[SemanticDiagnostic] = []
        self.scope: Scope | None = None

    def resolve(self, program: Program) -> None:
        # 1. Todas as assinaturas são coletadas antes dos corpos, o que
        #    permite chamadas antecipadas e recursão direta ou indireta.
        for function in program.functions:
            self.declare_function(function)

        # 2. main deve existir com a assinatura exata int main().
        self.check_main(program)

        # 3. Os corpos são percorridos em ordem de fonte, inclusive os de
        #    funções repetidas, para acumular erros independentes.
        for function in program.functions:
            self.resolve_function(function)

        if self.diagnostics:
            raise SemanticError(self.diagnostics)

    # ------------------------------------------------------------------
    # Funções

    def declare_function(self, function: FunctionDecl) -> None:
        symbol = FunctionSymbol(
            name=function.name,
            kind=SymbolKind.FUNCTION,
            type=function.return_type,
            declaration=function,
            parameter_types=tuple(p.type for p in function.parameters),
        )
        function.metadata["symbol"] = symbol
        if function.name in self.functions:
            # A primeira entrada da tabela global é preservada.
            self.error(
                SemanticErrorKind.DUPLICATE_FUNCTION,
                f"função '{function.name}' já foi declarada",
                function,
            )
            return
        self.functions[function.name] = symbol

    def check_main(self, program: Program) -> None:
        main = self.functions.get("main")
        if main is None:
            self.error(
                SemanticErrorKind.INVALID_MAIN,
                "o programa deve declarar a função int main()",
                program,
            )
        elif main.type is not TypeName.INT or main.parameter_types:
            self.error(
                SemanticErrorKind.INVALID_MAIN,
                "a função main deve ter a assinatura int main()",
                main.declaration,
            )

    def resolve_function(self, function: FunctionDecl) -> None:
        # O bloco externo da função cria o escopo dos parâmetros; por isso
        # eles não podem ser redeclarados diretamente no corpo.
        scope = Scope(parent=None)
        self.scope = scope
        for parameter in function.parameters:
            self.declare(parameter, parameter.name, parameter.type, SymbolKind.PARAMETER)
        self.resolve_block(function.body, scope)
        self.scope = None

    # ------------------------------------------------------------------
    # Escopos e declarações

    def declare(self, node: Node, name: str, type_: TypeName, kind: SymbolKind) -> None:
        assert self.scope is not None
        symbol = Symbol(name=name, kind=kind, type=type_, declaration=node)
        node.metadata["symbol"] = symbol
        if name in self.scope.symbols:
            self.error(
                SemanticErrorKind.DUPLICATE_DECLARATION,
                f"'{name}' já foi declarado neste escopo",
                node,
            )
            return
        self.scope.symbols[name] = symbol

    def resolve_block(self, block: Block, scope: Scope | None = None) -> None:
        if scope is None:
            scope = Scope(parent=self.scope)
        block.metadata["scope"] = scope
        previous = self.scope
        self.scope = scope
        for statement in block.statements:
            self.resolve_statement(statement)
        self.scope = previous

    # ------------------------------------------------------------------
    # Comandos

    def resolve_statement(self, statement: Stmt) -> None:
        if isinstance(statement, Block):
            self.resolve_block(statement)
        elif isinstance(statement, VarDecl):
            # A variável entra no escopo antes da visita do inicializador.
            self.declare(statement, statement.name, statement.type, SymbolKind.VARIABLE)
            if statement.initializer is not None:
                self.resolve_expression(statement.initializer)
        elif isinstance(statement, Assignment):
            self.resolve_expression(statement.target)
            self.resolve_expression(statement.value)
        elif isinstance(statement, CallStmt):
            self.resolve_expression(statement.call)
        elif isinstance(statement, IfStmt):
            self.resolve_expression(statement.condition)
            self.resolve_block(statement.then_block)
            if statement.else_block is not None:
                self.resolve_block(statement.else_block)
        elif isinstance(statement, WhileStmt):
            self.resolve_expression(statement.condition)
            self.resolve_block(statement.body)
        elif isinstance(statement, ReturnStmt):
            if statement.value is not None:
                self.resolve_expression(statement.value)
        elif isinstance(statement, PrintStmt):
            for item in statement.items:
                if not isinstance(item, StringLiteral):
                    self.resolve_expression(item)
        else:
            raise TypeError(f"comando inesperado: {type(statement).__name__}")

    # ------------------------------------------------------------------
    # Expressões

    def resolve_expression(self, expression: Expr) -> None:
        if isinstance(expression, IdentifierExpr):
            # Identificadores procuram somente nos escopos de variáveis.
            symbol = lookup(self.scope, expression.name)
            if symbol is None:
                self.error(
                    SemanticErrorKind.UNDECLARED_VARIABLE,
                    f"variável '{expression.name}' não foi declarada",
                    expression,
                )
            else:
                expression.metadata["symbol"] = symbol
        elif isinstance(expression, CallExpr):
            # Chamadas procuram somente na tabela global de funções.
            function = self.functions.get(expression.name)
            if function is None:
                self.error(
                    SemanticErrorKind.UNDECLARED_FUNCTION,
                    f"função '{expression.name}' não foi declarada",
                    expression,
                )
            else:
                expression.metadata["symbol"] = function
            for argument in expression.arguments:
                self.resolve_expression(argument)
        elif isinstance(expression, UnaryExpr):
            self.resolve_expression(expression.operand)
        elif isinstance(expression, BinaryExpr):
            self.resolve_expression(expression.left)
            self.resolve_expression(expression.right)
        elif isinstance(expression, (IntLiteral, BoolLiteral)):
            pass
        else:
            raise TypeError(f"expressão inesperada: {type(expression).__name__}")

    # ------------------------------------------------------------------

    def error(self, kind: SemanticErrorKind, message: str, node: Node) -> None:
        self.diagnostics.append(SemanticDiagnostic(kind, message, node.span))