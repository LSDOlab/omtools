from omtools.comps.matmat_comp import MatMatComp
from omtools.core.expression import Expression
from typing import List
import numpy as np


class matmat(Expression):
    def initialize(self, mat1, mat2):
        if isinstance(mat1, Expression) and isinstance(mat2, Expression):
            self.add_predecessor_node(mat1)
            self.add_predecessor_node(mat2)

            if mat1.shape[1] == mat2.shape[0] and len(mat2.shape) > 1:
                # Compute the output shape if both inputs are matrices
                self.shape = (mat1.shape[0], mat2.shape[1])

                self.build = lambda name: MatMatComp(
                    in_names=[mat1.name, mat2.name],
                    out_name=name,
                    in_shapes=[mat1.shape, mat2.shape],

                )
                         
            elif mat1.shape[1] == mat2.shape[0] and len(mat2.shape) == 1:
                self.shape = (mat1.shape[0], )

                mat2_shape = (mat2.shape[0], 1)

                self.build = lambda name: MatMatComp(
                    in_names=[mat1.name, mat2.name],
                    out_name=name,
                    in_shapes=[mat1.shape, mat2_shape],

                )
            else:
                raise Exception ("Cannot multiply: ", mat1.shape, "by", mat2.shape)


        else:
            raise TypeError(expr, " is not an Expression object")
