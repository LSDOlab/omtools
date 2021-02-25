from omtools.comps.tensor_inner_product_comp import TensorInnerProductComp
from omtools.comps.vector_inner_product_comp import VectorInnerProductComp

from omtools.core.variable import Variable
from typing import List
import numpy as np


def dot(expr1: Variable, expr2: Variable, axis=None):
    if not (isinstance(expr1, Variable) and isinstance(expr2, Variable)):
        raise TypeError("Arguments must both be Variable objects")
    out = Variable()
    out.add_dependency_node(expr1)
    out.add_dependency_node(expr2)

    if expr1.shape != expr2.shape:
        raise Exception("The shapes of the inputs must match!")

    if expr1.shape[axis] != 3:
        raise Exception(
            "The specified axis must correspond to the value of 3 in shape")

    if len(expr1.shape) == 1 and len(expr2.shape) == 1:
        out.build = lambda: VectorInnerProductComp(
            in_names=[expr1.name, expr2.name],
            out_name=out.name,
            in_shape=expr1.shape[0],
            in_vals=[expr1.val, expr2.val],
        )

    else:
        out.shape = np.delete(list(expr1.shape), axis)

        out.build = lambda: TensorDotProductComp(
            in_names=[expr1.name, expr2.name],
            out_name=out.name,
            in_shapes=[expr1.shape, expr2.shape],
            axis=axis,
            out_shape=out.shape,
            in_vals=[expr1.val, expr2.val],
        )
    return out
