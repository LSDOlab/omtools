from omtools.comps.single_tensor_sum_comp import SingleTensorSumComp
from omtools.comps.multiple_tensor_sum_comp import MultipleTensorSumComp
from omtools.core.expression import Expression
from typing import List
import numpy as np


def sum(*summands: List[Expression], axes=None):
    out = Expression()
    for expr in summands:
        if not isinstance(expr, Expression):
            raise TypeError(expr, " is not an Expression object")
        out.add_dependency_node(expr)

    if axes == None:
        if len(summands) == 1:
            out.build = lambda: SingleTensorSumComp(
                in_name=summands[0].name,
                shape=summands[0].shape,
                out_name=out.name,
                val=summands[0].val,
            )
        else:
            out.shape = expr.shape
            out.build = lambda: MultipleTensorSumComp(
                in_names=[expr.name for expr in summands],
                shape=expr.shape,
                out_name=out.name,
                vals=[expr.val for expr in summands],
            )
    else:
        output_shape = np.delete(expr.shape, axes)
        out.shape = tuple(output_shape)

        if len(summands) == 1:
            out.build = lambda: SingleTensorSumComp(
                in_name=expr.name,
                shape=expr.shape,
                out_name=out.name,
                out_shape=out.shape,
                axes=axes,
                val=summands[0].val,
            )
        else:
            out.build = lambda: MultipleTensorSumComp(
                in_names=[expr.name for expr in summands],
                shape=expr.shape,
                out_name=out.name,
                out_shape=out.shape,
                axes=axes,
                vals=[expr.val for expr in summands],
            )
    return out
