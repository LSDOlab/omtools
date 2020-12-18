from omtools.comps.coscomp import CosComp
from omtools.core.expression import Expression
from omtools.core.unary_function import UnaryFunction


class cos(UnaryFunction):
    def initialize(self, expr):
        if isinstance(expr, Expression):
            self.shape = expr.shape
            self.add_predecessor_node(expr)

            self.build = lambda name: CosComp(
                shape=expr.shape,
                in_name=expr.name,
                out_name=name,
            )

        else:
            raise TypeError(expr, " is not an Expression object")
