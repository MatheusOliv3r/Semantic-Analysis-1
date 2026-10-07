from __future__ import annotations

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
    PrintStmt,
    Program,
    ReturnStmt,
    Stmt,
    StringLiteral,
    TypeName,
    UnaryExpr,
    UnaryOperator,
    VarDecl,
    WhileStmt,
)
from semantic_errors import SemanticDiagnostic, SemanticError, SemanticErrorKind
from symbols import FunctionSymbol


# Estado interno de tipo desconhecido. Ele evita diagnósticos em cascata
# e nunca é gravado nos metadados, pois não pertence a TypeName.
UNKNOWN = None

MAX_INT_LITERAL = 2**63 - 1

ARITHMETIC = {
    BinaryOperator.ADD,
    BinaryOperator.SUBTRACT,
    BinaryOperator.MULTIPLY,
    BinaryOperator.DIVIDE,
    BinaryOperator.REMAINDER,
}
RELATIONAL = {
    BinaryOperator.LESS,
    BinaryOperator.LESS_EQUAL,
    BinaryOperator.GREATER,
    BinaryOperator.GREATER_EQUAL,
}
EQUALITY = {BinaryOperator.EQUAL, BinaryOperator.NOT_EQUAL}
LOGICAL = {BinaryOperator.LOGICAL_AND, BinaryOperator.LOGICAL_OR}


def check_types(program: Program) -> None:
    """Determine tipos de expressões e valide seus contextos."""

    TypeChecker().check(program)


class TypeChecker:
    def __init__(self) -> None:
        self.diagnostics: list[SemanticDiagnostic] = []
        self.current_function: FunctionDecl | None = None

    def check(self, program: Program) -> None:
        for function in program.functions:
            self.check_function(function)
        if self.diagnostics:
            raise SemanticError(self.diagnostics)

    # ------------------------------------------------------------------
    # Declarações

    def check_function(self, function: FunctionDecl) -> None:
        for parameter in function.parameters:
            if parameter.type is TypeName.VOID:
                self.error(
                    SemanticErrorKind.VOID_PARAMETER,
                    f"parâmetro '{parameter.name}' não pode ser void",
                    parameter,
                )
        self.current_function = function
        self.check_block(function.body)
        self.current_function = None

    def check_block(self, block: Block) -> None:
        for statement in block.statements:
            self.check_statement(statement)

    # ------------------------------------------------------------------
    # Comandos

    def check_statement(self, statement: Stmt) -> None:
        if isinstance(statement, Block):
            self.check_block(statement)
        elif isinstance(statement, VarDecl):
            self.check_var_decl(statement)
        elif isinstance(statement, Assignment):
            target = self.check_value(statement.target)
            value = self.check_value(statement.value)
            if not self.compatible(target, value):
                self.error(
                    SemanticErrorKind.ASSIGNMENT_TYPE_MISMATCH,
                    f"atribuição espera {target.value}, mas recebeu {value.value}",
                    statement.value,
                )
        elif isinstance(statement, CallStmt):
            # Chamadas int, bool e void podem ser comandos; o valor é descartado.
            self.check_expression(statement.call)
        elif isinstance(statement, IfStmt):
            self.check_condition(statement.condition)
            self.check_block(statement.then_block)
            if statement.else_block is not None:
                self.check_block(statement.else_block)
        elif isinstance(statement, WhileStmt):
            self.check_condition(statement.condition)
            self.check_block(statement.body)
        elif isinstance(statement, ReturnStmt):
            self.check_return(statement)
        elif isinstance(statement, PrintStmt):
            # print aceita strings e expressões int ou bool.
            for item in statement.items:
                if not isinstance(item, StringLiteral):
                    self.check_value(item)
        else:
            raise TypeError(f"comando inesperado: {type(statement).__name__}")

    def check_var_decl(self, declaration: VarDecl) -> None:
        if declaration.type is TypeName.VOID:
            self.error(
                SemanticErrorKind.VOID_VARIABLE,
                f"variável '{declaration.name}' não pode ser void",
                declaration,
            )
        if declaration.initializer is None:
            return
        value = self.check_value(declaration.initializer)
        expected = self.variable_type(declaration.type)
        if not self.compatible(expected, value):
            self.error(
                SemanticErrorKind.INITIALIZER_TYPE_MISMATCH,
                f"inicializador de '{declaration.name}' espera "
                f"{expected.value}, mas recebeu {value.value}",
                declaration.initializer,
            )

    def check_condition(self, condition: Expr) -> None:
        actual = self.check_value(condition)
        if actual is not UNKNOWN and actual is not TypeName.BOOL:
            self.error(
                SemanticErrorKind.CONDITION_TYPE_MISMATCH,
                f"condição deve ser bool, mas é {actual.value}",
                condition,
            )

    def check_return(self, statement: ReturnStmt) -> None:
        assert self.current_function is not None
        expected = self.current_function.return_type
        if statement.value is None:
            if expected is not TypeName.VOID:
                self.error(
                    SemanticErrorKind.RETURN_MISMATCH,
                    f"função '{self.current_function.name}' deve retornar "
                    f"{expected.value}",
                    statement,
                )
            return

        actual = self.check_value(statement.value)
        if expected is TypeName.VOID:
            self.error(
                SemanticErrorKind.RETURN_MISMATCH,
                f"função void '{self.current_function.name}' não pode "
                "retornar um valor",
                statement.value,
            )
        elif not self.compatible(expected, actual):
            self.error(
                SemanticErrorKind.RETURN_MISMATCH,
                f"retorno espera {expected.value}, mas recebeu {actual.value}",
                statement.value,
            )

    # ------------------------------------------------------------------
    # Expressões

    def check_value(self, expression: Expr) -> TypeName | None:
        """Visita uma expressão usada como valor; void não é permitido."""

        actual = self.check_expression(expression)
        if actual is TypeName.VOID:
            self.error(
                SemanticErrorKind.VOID_VALUE_USED,
                "chamada void não pode ser usada como valor",
                expression,
            )
            return UNKNOWN
        return actual

    def check_expression(self, expression: Expr) -> TypeName | None:
        result = self.compute_type(expression)
        if result is not UNKNOWN:
            expression.metadata["type"] = result
        return result

    def compute_type(self, expression: Expr) -> TypeName | None:
        if isinstance(expression, IntLiteral):
            if not 0 <= expression.value <= MAX_INT_LITERAL:
                self.error(
                    SemanticErrorKind.INTEGER_LITERAL_OUT_OF_RANGE,
                    f"literal inteiro {expression.value} fora do intervalo "
                    f"[0, {MAX_INT_LITERAL}]",
                    expression,
                )
            return TypeName.INT
        if isinstance(expression, BoolLiteral):
            return TypeName.BOOL
        if isinstance(expression, IdentifierExpr):
            # Variáveis e parâmetros void já foram reportados na declaração.
            return self.variable_type(expression.metadata["symbol"].type)
        if isinstance(expression, UnaryExpr):
            return self.check_unary(expression)
        if isinstance(expression, BinaryExpr):
            return self.check_binary(expression)
        if isinstance(expression, CallExpr):
            return self.check_call(expression)
        raise TypeError(f"expressão inesperada: {type(expression).__name__}")

    def check_unary(self, expression: UnaryExpr) -> TypeName:
        operand = self.check_value(expression.operand)
        if expression.operator is UnaryOperator.NEGATE:
            expected = TypeName.INT
        else:
            expected = TypeName.BOOL
        if operand is not UNKNOWN and operand is not expected:
            self.error(
                SemanticErrorKind.INVALID_UNARY_OPERAND,
                f"operador '{expression.operator.value}' exige {expected.value}, "
                f"mas recebeu {operand.value}",
                expression,
            )
        return expected

    def check_binary(self, expression: BinaryExpr) -> TypeName:
        # Os dois lados são sempre visitados para encontrar erros independentes.
        left = self.check_value(expression.left)
        right = self.check_value(expression.right)
        operator = expression.operator

        if operator in ARITHMETIC:
            operands, result = TypeName.INT, TypeName.INT
        elif operator in RELATIONAL:
            operands, result = TypeName.INT, TypeName.BOOL
        elif operator in LOGICAL:
            operands, result = TypeName.BOOL, TypeName.BOOL
        elif operator in EQUALITY:
            # == e != aceitam dois int ou dois bool.
            if left is not UNKNOWN and right is not UNKNOWN and left is not right:
                self.binary_error(expression, left, right)
            return TypeName.BOOL
        else:
            raise TypeError(f"operador inesperado: {operator}")

        if (left is not UNKNOWN and left is not operands) or (
            right is not UNKNOWN and right is not operands
        ):
            self.binary_error(expression, left, right)
        return result

    def binary_error(
        self, expression: BinaryExpr, left: TypeName | None, right: TypeName | None
    ) -> None:
        self.error(
            SemanticErrorKind.INVALID_BINARY_OPERANDS,
            f"operador '{expression.operator.value}' não aceita "
            f"{self.describe(left)} e {self.describe(right)}",
            expression,
        )

    def check_call(self, call: CallExpr) -> TypeName:
        function: FunctionSymbol = call.metadata["symbol"]
        # Todos os argumentos são visitados, mesmo com aridade errada.
        arguments = [self.check_value(argument) for argument in call.arguments]

        if len(arguments) != len(function.parameter_types):
            self.error(
                SemanticErrorKind.ARITY_MISMATCH,
                f"função '{call.name}' espera {len(function.parameter_types)} "
                f"argumento(s), mas recebeu {len(arguments)}",
                call,
            )

        for argument, actual, expected in zip(
            call.arguments, arguments, function.parameter_types
        ):
            expected_type = self.variable_type(expected)
            if not self.compatible(expected_type, actual):
                self.error(
                    SemanticErrorKind.ARGUMENT_TYPE_MISMATCH,
                    f"argumento de '{call.name}' espera {expected_type.value}, "
                    f"mas recebeu {actual.value}",
                    argument,
                )
        return function.type

    # ------------------------------------------------------------------
    # Auxiliares

    @staticmethod
    def variable_type(declared: TypeName) -> TypeName | None:
        """Variáveis e parâmetros void são inválidos e ficam desconhecidos."""

        return UNKNOWN if declared is TypeName.VOID else declared

    @staticmethod
    def compatible(expected: TypeName | None, actual: TypeName | None) -> bool:
        # Não há coerções implícitas: os tipos devem ser exatamente iguais.
        return expected is UNKNOWN or actual is UNKNOWN or expected is actual

    @staticmethod
    def describe(type_: TypeName | None) -> str:
        return "tipo desconhecido" if type_ is UNKNOWN else type_.value

    def error(self, kind: SemanticErrorKind, message: str, node: Node) -> None:
        self.diagnostics.append(SemanticDiagnostic(kind, message, node.span))