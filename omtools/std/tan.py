from omtools.comps.tancomp import TanComp
from omtools.core.expression import Expression


class tan(Expression):
    def initialize(self, expr):
        if isinstance(expr, Expression):
            self.shape = expr.shape
            self.add_predecessor_node(expr)

            self.build = lambda: TanComp(
                shape=expr.shape,
                in_name=expr.name,
                out_name=self.name,
            )

        else:
            raise TypeError(expr, " is not an Expression object")
