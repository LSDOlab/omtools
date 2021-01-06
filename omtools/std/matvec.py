from omtools.comps.matvec_comp import MatVecComp
from omtools.core.expression import Expression
from typing import List
import numpy as np


class matvec(Expression):
    def initialize(self, mat1, vec1):
        if isinstance(mat1, Expression) and isinstance(vec1, Expression):
            self.add_predecessor_node(mat1)
            self.add_predecessor_node(vec1)

            if mat1.shape[1] == vec1.shape[0] and len(vec1.shape) == 1:

                self.shape = (mat1.shape[0], )

                self.build = lambda name: MatVecComp(
                    in_names=[mat1.name, vec1.name],
                    out_name=name,
                    in_shapes=[mat1.shape, vec1.shape],

                )
                         
            else:
                raise Exception ("Cannot multiply: ", mat1.shape, "by", vec1.shape)


        else:
            raise TypeError(expr, " is not an Expression object")