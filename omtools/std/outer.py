from omtools.comps.tensor_outer_product_comp import TensorOuterProductComp
from omtools.comps.vector_outer_product_comp import VectorOuterProductComp

from omtools.core.expression import Expression
from typing import List
import numpy as np


class outer(Expression):
    def initialize(self, expr1: Expression, expr2: Expression):

        if isinstance(expr1, Expression) == False:
            raise TypeError(expr1, " is not an Expression object")
        elif isinstance(expr2, Expression) == False:
            raise TypeError(expr2, " is not an Expression object")

        self.add_predecessor_node(expr1)
        self.add_predecessor_node(expr2)

        if len(expr1.shape) == 1 and len(expr2.shape) == 1:
            self.shape = tuple(list(expr1.shape) + list(expr2.shape))

            self.build = lambda: VectorOuterProductComp(
                in_names=[expr1.name, expr2.name],
                out_name=self.name,
                in_shapes=[expr1.shape[0], expr2.shape[0]],
            )

        else:
            self.shape = tuple(list(expr1.shape) + list(expr2.shape))

            self.build = lambda: TensorOuterProductComp(
                in_names=[expr1.name, expr2.name],
                out_name=self.name,
                in_shapes=[expr1.shape, expr2.shape],
            )
