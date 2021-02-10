from omtools.comps.matvec_comp import MatVecComp
from omtools.core.expression import Expression
from typing import List
import numpy as np


def matvec(mat1, vec1):
    if not (isinstance(mat1, Expression) and isinstance(vec1, Expression)):
        raise TypeError("Arguments must both be Expression objects")
    out = Expression()
    out.add_dependency_node(mat1)
    out.add_dependency_node(vec1)

    if mat1.shape[1] == vec1.shape[0] and len(vec1.shape) == 1:

        out.shape = (mat1.shape[0], )

        out.build = lambda: MatVecComp(
            in_names=[mat1.name, vec1.name],
            out_name=out.name,
            in_shapes=[mat1.shape, vec1.shape],
        )

    else:
        raise Exception("Cannot multiply: ", mat1.shape, "by", vec1.shape)
    return out
