from omtools.comps.vectorized_pnorm_comp import VectorizedPnormComp
from omtools.comps.vectorized_axiswise_pnorm_comp import VectorizedAxisWisePnormComp
from omtools.core.expression import Expression
from typing import List
import numpy as np


class pnorm(Expression):
    def initialize(self, expr, pnorm_type, axis = None):
        if isinstance(expr, Expression):
            self.add_predecessor_node(expr)

            if pnorm_type%2 != 0 or pnorm_type <= 0:
                raise Exception(pnorm_type, " is not positive OR is not even")

            else:
                if axis == None:
                    self.build = lambda name: VectorizedPnormComp(
                        shape = expr.shape,
                        in_name = expr.name,
                        out_name = name,
                        pnorm_type = pnorm_type,
                    )
                else:
                    output_shape = np.delete(expr.shape, axis)
                    self.shape = tuple(output_shape)

                    self.build = lambda name: VectorizedAxisWisePnormComp(
                        shape = expr.shape,
                        in_name = expr.name,
                        out_shape = self.shape,
                        out_name = name,
                        pnorm_type = pnorm_type,
                        axis = axis,
                    )               

        else:
            raise TypeError(expr, " is not an Expression object")
