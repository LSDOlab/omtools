from omtools.comps.reshape_comp import ReshapeComp
from omtools.core.variable import Variable


def reshape(expr, new_shape):
    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")
    out = Variable()
    out.shape = new_shape
    out.add_dependency_node(expr)
    out.build = lambda: ReshapeComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
        new_shape=out.shape,
        val=expr.val,
    )
    return out
