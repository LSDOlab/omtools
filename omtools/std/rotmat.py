from omtools.comps.rotation_matrix_comp import RotationMatrixComp
from omtools.core.expression import Expression


def rotmat(angle, axis: str):
    if not isinstance(angle, Expression):
        raise TypeError(angle, " is not an Expression object")
    out = Expression()
    out.add_dependency_node(angle)

    if angle.shape == (1, ):
        out.shape = (3, 3)

    else:
        out.shape = angle.shape + (3, 3)

    out.build = lambda: RotationMatrixComp(
        shape=angle.shape,
        in_name=angle.name,
        out_name=out.name,
        axis=axis,
    )
    return out
