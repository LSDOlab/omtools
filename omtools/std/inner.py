from omtools.comps.tensor_inner_product_comp import TensorInnerProductComp
from omtools.comps.vector_inner_product_comp import VectorInnerProductComp

from omtools.core.expression import Expression
from typing import List
import numpy as np


def inner(expr1: Expression, expr2: Expression, axes=None):
    if not isinstance(expr1, Expression):
        raise TypeError(expr1, " is not an Expression object")
    elif not isinstance(expr2, Expression):
        raise TypeError(expr2, " is not an Expression object")

    out = Expression()
    out.add_dependency_node(expr1)
    out.add_dependency_node(expr2)

    if len(expr1.shape) == 1 and len(expr2.shape) == 1:
        out.build = lambda: VectorInnerProductComp(
            in_names=[expr1.name, expr2.name],
            out_name=out.name,
            in_shape=expr1.shape[0],
            in_vals=[expr1.val, expr2.val],
        )

    else:
        new_in0_shape = np.delete(list(expr1.shape), axes[0])
        new_in1_shape = np.delete(list(expr2.shape), axes[1])
        out.shape = tuple(np.append(new_in0_shape, new_in1_shape))

        out.build = lambda: TensorInnerProductComp(
            in_names=[expr1.name, expr2.name],
            out_name=out.name,
            in_shapes=[expr1.shape, expr2.shape],
            axes=axes,
            out_shape=out.shape,
            in_vals=[expr1.val, expr2.val],
        )
    return out
