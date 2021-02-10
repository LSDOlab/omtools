from omtools.comps.matmat_comp import MatMatComp
from omtools.core.expression import Expression
from typing import List
import numpy as np


def matmat(mat1, mat2):
    if not (isinstance(mat1, Expression) and isinstance(mat2, Expression)):
        raise TypeError("Arguments must both be Expression objects")
    out = Expression()
    out.add_dependency_node(mat1)
    out.add_dependency_node(mat2)

    if mat1.shape[1] == mat2.shape[0] and len(mat2.shape) == 2:
        # Compute the output shape if both inputs are matrices
        out.shape = (mat1.shape[0], mat2.shape[1])

        out.build = lambda: MatMatComp(
            in_names=[mat1.name, mat2.name],
            out_name=out.name,
            in_shapes=[mat1.shape, mat2.shape],
        )

    elif mat1.shape[1] == mat2.shape[0] and len(mat2.shape) == 1:
        out.shape = (mat1.shape[0], 1)

        mat2_shape = (mat2.shape[0], 1)

        out.build = lambda: MatMatComp(
            in_names=[mat1.name, mat2.name],
            out_name=out.name,
            in_shapes=[mat1.shape, mat2_shape],
        )
    else:
        raise Exception("Cannot multiply: ", mat1.shape, "by", mat2.shape)
    return out
