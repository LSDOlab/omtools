from omtools.comps.single_tensor_sum_comp import SingleTensorSumComp
from omtools.comps.multiple_tensor_sum_comp import MultipleTensorSumComp
from omtools.core.expression import Expression
from typing import List
import numpy as np

class sum(Expression):
    def initialize(self, *summands: List[Expression], axes = None):   
        for expr in summands:
            if isinstance(expr, Expression) == False:
                raise TypeError(expr, " is not an Expression object")

            self.add_predecessor_node(expr)

        if axes == None:
            if len(summands) == 1:
                self.build = lambda name: SingleTensorSumComp(
                    in_name=expr.name,
                    shape = expr.shape,
                    out_name = name,
                )
            else:
                self.shape = expr.shape
                self.build = lambda name: MultipleTensorSumComp(
                    in_names=[expr.name for expr in summands],
                    shape = expr.shape,
                    out_name = name,
                )
        else:
            output_shape = np.delete(expr.shape, axes)
            self.shape = tuple(output_shape)

            if len(summands) == 1:
                self.build = lambda name: SingleTensorSumComp(
                    in_name=expr.name,
                    shape = expr.shape,
                    out_name = name,
                    out_shape = self.shape,
                    axes = axes,
                )
            else:
                self.build = lambda name: MultipleTensorSumComp(
                    in_names=[expr.name for expr in summands],
                    shape = expr.shape,
                    out_name = name,
                    out_shape = self.shape,
                    axes = axes,
                )            