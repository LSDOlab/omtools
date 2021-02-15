from omtools.comps.vectorized_pnorm_comp import VectorizedPnormComp
from omtools.comps.vectorized_axiswise_pnorm_comp import VectorizedAxisWisePnormComp
from omtools.core.expression import Expression
from typing import List
import numpy as np


def pnorm(expr, pnorm_type, axis=None):
    if not isinstance(expr, Expression):
        raise TypeError(expr, " is not an Expression object")
    out = Expression()
    out.add_dependency_node(expr)

    if pnorm_type % 2 != 0 or pnorm_type <= 0:
        raise Exception(pnorm_type, " is not positive OR is not even")

    else:
        if axis == None:
            out.build = lambda: VectorizedPnormComp(
                shape=expr.shape,
                in_name=expr.name,
                out_name=out.name,
                pnorm_type=pnorm_type,
                val=expr.val,
            )
        else:
            output_shape = np.delete(expr.shape, axis)
            out.shape = tuple(output_shape)

            out.build = lambda: VectorizedAxisWisePnormComp(
                shape=expr.shape,
                in_name=expr.name,
                out_shape=out.shape,
                out_name=out.name,
                pnorm_type=pnorm_type,
                axis=axis,
                val=expr.val,
            )
    return out
