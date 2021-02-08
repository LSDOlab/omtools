from omtools.comps.transpose_comp import TransposeComp
from omtools.core.expression import Expression
from typing import List
import numpy as np


class transpose(Expression):
    def initialize(self, expr: Expression):

        if isinstance(expr, Expression) == False:
            raise TypeError(expr, " is not an Expression object")

        self.add_predecessor_node(expr)

        self.shape = expr.shape[::-1]
        self.build = lambda: TransposeComp(
            in_name=expr.name,
            in_shape=expr.shape,
            out_name=self.name,
            out_shape=self.shape,
        )
