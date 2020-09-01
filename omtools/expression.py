import numbers

from openmdao.api import Group
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

# TODO: check that naming conventions are robust to duplication of
# components and builders
# TODO: group expressions by parentheses
# TODO: guarantee that explicit components don't form feedback loops if
# there aren't any.


class Expression(object):
    """
    Store builders for creating components that evaluate expressions
    and define operations on a Expression objects
    """
    def __init__(self, *args, **kwargs):
        # name of component, default name for output
        self.name = ""
        self.shape = ()
        self.input_names = []
        self.input_shapes = []
        # Storing builders in a dict avoids duplication of components
        # NOTE: requires careful naming of components
        self.builders = {}
        self.initialize(*args, **kwargs)

    def initialize(self):
        pass

    def rename(self,name):
        """
        Use this method to rename output variables for access outside
        the group where expressions are defined
        """
        self.builders[self.name].name = name

    def __add__(self, other):
        return BinaryExpression(self, other, Plus(self, other))

    def __sub__(self, other):
        return BinaryExpression(self, other, Minus(self, other))

    def __mul__(self, other):
        return BinaryExpression(self, other, Times(self, other))

    def __pow__(self, other):
        return BinaryExpression(self, other, Pow(self, other))


class Variable(Expression):
    """
    Container for declaring an input or an output
    """
    def initialize(self, name, shape=(1, )):
        self.name = name
        self.shape = shape

    def __repr__(self):
        ns = "Variable "
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

        # TODO: supply builders with literals
        if isinstance(expr2, Expression) and isinstance(expr2, Expression):
            self.builders = {
                **expr1.builders,
                **expr2.builders,
            }
        elif isinstance(expr2, Expression) and isinstance(
                expr2, numbers.Number):
            self.builders = {
                **expr1.builders,
            }
        elif isinstance(expr1, numbers.Number) and isinstance(
                expr2, Expression):
            self.builders = {
                **expr2.builders,
            }
        self.builders[builder.name] = builder

    def __repr__(self):
        ns = "BinaryExpression "
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
            shape_str += ")"
        n = "('" + self.name + "', " + shape_str + ")"
        ns = ns + n
        return ns
