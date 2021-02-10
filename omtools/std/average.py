from omtools.comps.single_tensor_average_comp import SingleTensorAverageComp
from omtools.comps.multiple_tensor_average_comp import MultipleTensorAverageComp
from omtools.core.expression import Expression
from typing import List
import numpy as np


def average(*operands: List[Expression], axes=None):
    out = Expression()
    for expr in operands:
        if not isinstance(expr, Expression):
            raise TypeError(expr, " is not an Expression object")
        out.add_dependency_node(expr)

    if axes == None:
        if len(operands) == 1:
            out.build = lambda: SingleTensorAverageComp(
                in_name=expr.name,
                shape=expr.shape,
                out_name=out.name,
            )
        else:
            out.shape = expr.shape
            out.build = lambda: MultipleTensorAverageComp(
                in_names=[expr.name for expr in operands],
                shape=expr.shape,
                out_name=out.name,
            )
    else:
        output_shape = np.delete(expr.shape, axes)
        out.shape = tuple(output_shape)

        if len(operands) == 1:
            out.build = lambda: SingleTensorAverageComp(
                in_name=expr.name,
                shape=expr.shape,
                out_name=out.name,
                out_shape=out.shape,
                axes=axes,
            )
        else:
            out.build = lambda: MultipleTensorAverageComp(
                in_names=[expr.name for expr in operands],
                shape=expr.shape,
                out_name=out.name,
                out_shape=out.shape,
                axes=axes,
            )
    return out
