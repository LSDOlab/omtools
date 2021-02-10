from omtools.comps.reorder_axes_comp import ReorderAxesComp
from omtools.core.expression import Expression
from typing import List
import numpy as np
from omtools.utils.reorder_axes_utils import compute_new_axes_locations


def reorder_axes(expr: Expression, operation: str):
    if not isinstance(expr, Expression):
        raise TypeError(expr, " is not an Expression object")
    out = Expression()
    out.add_dependency_node(expr)

    # Computing out_shape
    new_axes_locations = compute_new_axes_locations(expr.shape, operation)
    out.shape = tuple(expr.shape[i] for i in new_axes_locations)

    out.build = lambda: ReorderAxesComp(
        in_name=expr.name,
        in_shape=expr.shape,
        out_name=out.name,
        out_shape=out.shape,
        operation=operation,
        new_axes_locations=new_axes_locations,
    )
    return out
