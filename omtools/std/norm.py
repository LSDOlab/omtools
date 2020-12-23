from omtools.comps.norm_comp import NormComp
from omtools.core.expression import Expression
from typing import List


class norm(Expression):
    def initialize(self, expr, norm_type: str, axis = None):
        if isinstance(expr, Expression):
            self.shape = expr.shape
            self.add_predecessor_node(expr)

            self.build = lambda name: NormComp(
                shape = expr.shape,
                in_name = expr.name,
                out_name = name,
                norm_type = norm_type,
                axis = axis,
            )

        else:
            raise TypeError(expr, " is not an Expression object")
