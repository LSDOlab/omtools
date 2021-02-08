from omtools.comps.tensor_inner_product_comp import TensorInnerProductComp
from omtools.comps.vector_inner_product_comp import VectorInnerProductComp

from omtools.core.expression import Expression
from typing import List
import numpy as np


class dot(Expression):
    def initialize(self, expr1: Expression, expr2: Expression, axis=None):
        if isinstance(expr1, Expression) and isinstance(expr2, Expression):

            self.add_predecessor_node(expr1)
            self.add_predecessor_node(expr2)

            if expr1.shape != expr2.shape:
                raise Exception("The shapes of the inputs must match!")
            else:
                self.shape = expr1.shape

        if len(expr1.shape) == 1 and len(expr2.shape) == 1:
            self.build = lambda: VectorInnerProductComp(
                in_names=[expr1.name, expr2.name],
                out_name=self.name,
                in_shape=expr1.shape[0],
            )

        else:
            new_in1_shape = np.delete(list(expr1.shape), axis)
            new_in2_shape = np.delete(list(expr2.shape), axis)
            self.shape = tuple(np.append(new_in1_shape, new_in2_shape))

            self.build = lambda: TensorInnerProductComp(
                in_names=[expr1.name, expr2.name],
                out_name=self.name,
                in_shapes=[expr1.shape, expr2.shape],
                axes=([axis], [axis]),
                out_shape=self.shape,
            )
