import numbers

from openmdao.api import Group, NewtonSolver
from openmdao.core.component import Component

from lsdo_utils.api import LinearCombinationComp, PowerCombinationComp
from omtools.builders.binops import Minus, Plus, Pow, Times
"""
Expression classes

Expressions don't contain components; they contain instructions for
constructing components. Duplicate instructions are automatically
removed to avoid creating components with naming conflicts and
redundant computations.

Variables are nullary expressions, with no arguments.
Special methods return binary expressions.
Unary expressions are defined elsewhere.
"""


class Expression(object):
    """
    Container for instructions that direct OpenMDAO to create components
    that OpenMDAO uses to evaluate user defined expressions
    """
    def __init__(self, *args, **kwargs):
        # name of component, default name for output
        self.name = ""
        self.shape = ()
        self.input_names = []
        self.input_shapes = []
        # Storing builders in a dict avoids duplication of components
        # NOTE: requires careful naming of components
        self.builders = []
        self.initialize(*args, **kwargs)

    def initialize(self):
        pass

    def rename(self, name):
        """
        Use this method to rename output variables for access outside
        the group where expressions are defined
        """

    # TODO: accept arguments of number type
    def __add__(self, other):
        return BinaryExpression(self, other, Plus(self, other))

    def __sub__(self, other):
        return BinaryExpression(self, other, Minus(self, other))

    def __mul__(self, other):
        return BinaryExpression(self, other, Times(self, other))

    def __pow__(self, other):
        return BinaryExpression(self, other, Pow(self, other))


class NullaryExpression(Expression):
    """
    Container for declaring an input or an output
    """
    def initialize(self, name, shape=(1, )):
        self.name = name
        self.shape = shape

    def __getitem__(self, key):
        pass


class Indep(NullaryExpression):
    """
    Container for creating a constant or design variable
    """
    def __repr__(self):
        ns = "Independent Variable "
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        ns += "('" + self.name + "', " + shape_str + ")"
        return ns


class Input(NullaryExpression):
    """
    Container for declaring an input
    """
    def __repr__(self):
        ns = "Input Variable "
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        ns += "('" + self.name + "', " + shape_str + ")"
        return ns


class ExplicitOutput(NullaryExpression):
    """
    Container for declaring an explicit output
    """
    def initialize(self, group, name, shape=(1, )):
        self.name = name
        self.shape = shape
        self.group = group

    def define(self, expr):
        self.group.register_output(self.name, expr)

    # TODO: add slices
    def __setitem__(self, key, value):
        pass

    def __repr__(self):
        ns = "Explicit Output "
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        ns += "('" + self.name + "', " + shape_str + ")"
        return ns


class ImplicitOutput(NullaryExpression):
    """
    Container for declaring an impicit output
    """
    def initialize(self, group, name, shape=(1, )):
        self.name = name
        self.shape = shape
        self.group = group

    def define_residual(self, expr):
        self.group.register_output(self.name, expr)

    def __repr__(self):
        ns = "Implicit Output "
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        ns += "('" + self.name + "', " + shape_str + ")"
        return ns


class UnaryExpression(Expression):
    def initialize(self, expr):
        self.name = "UnaryExpression " + expr.name
        self.input_names = [expr.name]
        self.input_shapes = [expr.shape]
        self.builders = expr.builders

    def __repr__(self):
        ns = self.name
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        ns += "('" + self.name + "', " + shape_str + ")"
        return ns


class BinaryExpression(Expression):
    def initialize(self, expr1, expr2, builder, **kwargs):
        self.name = builder.name
        self.shape = builder.shape

        # TODO: supply builders with literals/number types
        if isinstance(expr2, Expression) and isinstance(expr2, Expression):
            self.builders = [
                *expr1.builders,
                *expr2.builders,
            ]
        elif isinstance(expr2, Expression) and isinstance(
                expr2, numbers.Number):
            self.builders = [
                *expr1.builders,
            ]
        elif isinstance(expr1, numbers.Number) and isinstance(
                expr2, Expression):
            self.builders = [
                *expr2.builders,
            ]
        self.builders.append(builder)

    def __repr__(self):
        ns = "BinaryExpression "
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
            shape_str += ")"
        n = "('" + self.name + "', " + shape_str + ")"
        ns = ns + n
        return ns
