from omtools.comps.seccomp import SecComp
from omtools.core.variable import Variable


def sec(expr):
    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")
    out = Variable()
    out.shape = expr.shape
    out.add_dependency_node(expr)
    out.build = lambda: SecComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
        val=expr.val,
    )
    return out
