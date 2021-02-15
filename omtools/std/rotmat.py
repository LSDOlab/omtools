from omtools.comps.rotation_matrix_comp import RotationMatrixComp
from omtools.core.variable import Variable


def rotmat(expr: Variable, axis: str):
    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")
    out = Variable()
    out.add_dependency_node(expr)

    if expr.shape == (1, ):
        out.shape = (3, 3)

    else:
        out.shape = expr.shape + (3, 3)

    out.build = lambda: RotationMatrixComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
        axis=axis,
        val=expr.val,
    )
    return out
