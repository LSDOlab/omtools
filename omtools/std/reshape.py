from omtools.comps.reshape_comp import ReshapeComp
from omtools.core.expression import Expression


def reshape(expr, new_shape):
    if not isinstance(expr, Expression):
        raise TypeError(expr, " is not an Expression object")
    out = Expression()
    out.shape = new_shape
    out.add_dependency_node(expr)
    out.build = lambda: ReshapeComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
        new_shape=out.shape,
    )
    return out
