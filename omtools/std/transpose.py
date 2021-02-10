from omtools.comps.transpose_comp import TransposeComp
from omtools.core.expression import Expression
from typing import List
import numpy as np


def transpose(expr: Expression):
    if not isinstance(expr, Expression):
        raise TypeError(expr, " is not an Expression object")
    out = Expression()
    out.add_dependency_node(expr)
    out.shape = expr.shape[::-1]
    out.build = lambda: TransposeComp(
        in_name=expr.name,
        in_shape=expr.shape,
        out_name=out.name,
        out_shape=out.shape,
    )
    return out
