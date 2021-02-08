from omtools.comps.coscomp import CosComp
from omtools.core.expression import Expression


class cos(Expression):
    def initialize(self, expr):
        if isinstance(expr, Expression):
            self.shape = expr.shape
            self.add_predecessor_node(expr)

            self.build = lambda: CosComp(
                shape=expr.shape,
                in_name=expr.name,
                out_name=self.name,
            )

        else:
            raise TypeError(expr, " is not an Expression object")
