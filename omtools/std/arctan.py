from omtools.comps.arctancomp import ArctanComp
from omtools.core.variable import Variable


def arctan(expr):
    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")
    out = Variable()
    out.shape = expr.shape
    out.add_dependency_node(expr)

    out.build = lambda: ArctanComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
        val=expr.val,
    )
    return out
