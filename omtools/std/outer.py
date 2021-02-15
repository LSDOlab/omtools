from omtools.comps.tensor_outer_product_comp import TensorOuterProductComp
from omtools.comps.vector_outer_product_comp import VectorOuterProductComp

from omtools.core.expression import Expression
from typing import List
import numpy as np


def outer(expr1: Expression, expr2: Expression):
    if not isinstance(expr1, Expression):
        raise TypeError(expr1, " is not an Expression object")
    elif not isinstance(expr2, Expression):
        raise TypeError(expr2, " is not an Expression object")
    out = Expression()
    out.add_dependency_node(expr1)
    out.add_dependency_node(expr2)

    if len(expr1.shape) == 1 and len(expr2.shape) == 1:
        out.shape = tuple(list(expr1.shape) + list(expr2.shape))

        out.build = lambda: VectorOuterProductComp(
            in_names=[expr1.name, expr2.name],
            out_name=out.name,
            in_shapes=[expr1.shape[0], expr2.shape[0]],
            in_vals=[expr1.val, expr2.val],
        )

    else:
        out.shape = tuple(list(expr1.shape) + list(expr2.shape))

        out.build = lambda: TensorOuterProductComp(
            in_names=[expr1.name, expr2.name],
            out_name=out.name,
            in_shapes=[expr1.shape, expr2.shape],
            in_vals=[expr1.val, expr2.val],
        )
    return out
