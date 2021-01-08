from omtools.comps.tensor_inner_product_comp import TensorInnerProductComp
from omtools.comps.vector_inner_product_comp import VectorInnerProductComp

from omtools.core.expression import Expression
from typing import List
import numpy as np

class dot(Expression):
    def initialize(self, expr1: Expression, expr2: Expression, axes=None):   

        if isinstance(expr1, Expression) == False:
            raise TypeError(expr1, " is not an Expression object")
        elif isinstance(expr2, Expression) == False:
            raise TypeError(expr2, " is not an Expression object")

        self.add_predecessor_node(expr1)
        self.add_predecessor_node(expr2)

        if len(expr1.shape) == 1 and len(expr2.shape) == 1:
            self.build = lambda name: VectorInnerProductComp(
                in_names=[expr1.name, expr2.name],
                out_name= name,
                in_shape= expr1.shape[0],
            )
        
        else:
            new_in0_shape = np.delete(list(expr1.shape), axes[0])
            new_in1_shape = np.delete(list(expr2.shape), axes[1])
            self.shape = tuple(np.append(new_in0_shape, new_in1_shape))
        
            self.build = lambda name: TensorInnerProductComp(
                    in_names=[expr1.name, expr2.name],
                    out_name= name,
                    in_shapes= [expr1.shape, expr2.shape],
                    axes= axes,
                    out_shape = self.shape,
                )
            

        