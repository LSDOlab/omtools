from omtools.comps.axiswise_max_comp import AxisMaxComp
from omtools.comps.elementwise_max_comp import ElementwiseMaxComp
from omtools.comps.scalar_extremum_comp import ScalarExtremumComp

from omtools.core.expression import Expression

import numpy as np


class max(Expression):
    def initialize(self, *exprs, axis=None, rho=20.):
        for expr in exprs:
            if isinstance(expr, Expression):
                self.add_predecessor_node(expr)
            else:
                raise TypeError(expr, " is not an Expression object")

        if len(exprs) == 1 and axis != None:
            output_shape = np.delete(expr.shape, axis)
            self.shape = tuple(output_shape)

            self.build = lambda: AxisMaxComp(
                shape=expr.shape,
                in_name=expr.name,
                axis=axis,
                out_name=self.name,
                rho=rho,
            )

        elif len(exprs) > 1 and axis == None:

            shape = exprs[0].shape
            for expr in exprs:
                if shape != expr.shape:
                    raise Exception("The shapes of the inputs must match!")

            self.shape = expr.shape

            self.build = lambda: ElementwiseMaxComp(
                shape=expr.shape,
                in_names=[expr.name for expr in exprs],
                out_name=self.name,
                rho=rho,
            )

        elif len(exprs) == 1 and axis == None:

            self.build = lambda: ScalarExtremumComp(
                shape=expr.shape,
                in_name=expr.name,
                out_name=self.name,
                rho=rho,
                lower_flag=False,
            )

        else:
            raise Exception("Do not give multiple inputs and an axis")
