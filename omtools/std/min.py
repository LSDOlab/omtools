from omtools.comps.axiswise_min_comp import AxisMinComp
from omtools.comps.elementwise_min_comp import ElementwiseMinComp
from omtools.comps.scalar_extremum_comp import ScalarExtremumComp

from omtools.core.expression import Expression

import numpy as np


def min(*exprs, axis=None, rho=20.):
    out = Expression()
    for expr in exprs:
        if not isinstance(expr, Expression):
            raise TypeError(expr, " is not an Expression object")
        out.add_dependency_node(expr)

    if len(exprs) == 1 and axis != None:
        output_shape = np.delete(expr.shape, axis)
        out.shape = tuple(output_shape)

        out.build = lambda: AxisMinComp(
            shape=expr.shape,
            in_name=expr.name,
            axis=axis,
            out_name=out.name,
            rho=rho,
        )

    elif len(exprs) > 1 and axis == None:

        shape = exprs[0].shape
        for expr in exprs:
            if shape != expr.shape:
                raise Exception("The shapes of the inputs must match!")

        out.shape = expr.shape

        out.build = lambda: ElementwiseMinComp(
            shape=expr.shape,
            in_names=[expr.name for expr in exprs],
            out_name=out.name,
            rho=rho,
        )

    elif len(exprs) == 1 and axis == None:

        out.build = lambda: ScalarExtremumComp(
            shape=expr.shape,
            in_name=expr.name,
            out_name=out.name,
            rho=rho,
            lower_flag=True,
        )

    else:
        raise Exception("Do not give multiple inputs and an axis")
    return out
