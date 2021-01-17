from omtools.comps.single_tensor_average_comp import SingleTensorAverageComp
from omtools.comps.multiple_tensor_average_comp import MultipleTensorAverageComp
from omtools.core.expression import Expression
from typing import List
import numpy as np

class average(Expression):
    def initialize(self, *operands: List[Expression], axes = None):
     
        for expr in operands:
            if isinstance(expr, Expression) == False:
                raise TypeError(expr, " is not an Expression object")

            self.add_predecessor_node(expr)

        if axes == None:
            if len(operands) == 1:
                self.build = lambda name: SingleTensorAverageComp(
                    in_name=expr.name,
                    shape = expr.shape,
                    out_name = name,
                )
            else:
                self.shape = expr.shape
                self.build = lambda name: MultipleTensorAverageComp(
                    in_names=[expr.name for expr in operands],
                    shape = expr.shape,
                    out_name = name,
                )
        else:
            output_shape = np.delete(expr.shape, axes)
            self.shape = tuple(output_shape)

            if len(operands) == 1:
                self.build = lambda name: SingleTensorAverageComp(
                    in_name=expr.name,
                    shape = expr.shape,
                    out_name = name,
                    out_shape = self.shape,
                    axes = axes,
                )
            else:
                self.build = lambda name: MultipleTensorAverageComp(
                    in_names=[expr.name for expr in operands],
                    shape = expr.shape,
                    out_name = name,
                    out_shape = self.shape,
                    axes = axes,
                )   