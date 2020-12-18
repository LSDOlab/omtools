import numbers

from omtools.comps.sincomp import SinComp
from omtools.core.expression import Expression
from omtools.core.indep import Indep
from omtools.core.unary_function import UnaryFunction


class sin(UnaryFunction):
    def initialize(self, expr):
        if isinstance(expr, Expression):
            self.shape = expr.shape
            self.add_predecessor_node(expr)

            self.build = lambda name: SinComp(
                shape=expr.shape,
                in_name=expr.name,
                out_name=name,
            )

        else:
            raise TypeError(expr, " is not an Expression object")
