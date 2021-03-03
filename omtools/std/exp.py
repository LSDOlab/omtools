from omtools.comps.expcomp import ExpComp
from omtools.core.variable import Variable


def exp(expr):
    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")
    out = Variable()
    out.shape = expr.shape
    out.add_dependency_node(expr)
    out.build = lambda: ExpComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
        val=expr.val,
    )
    return out
