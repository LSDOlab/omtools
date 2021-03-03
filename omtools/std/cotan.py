from omtools.comps.cotancomp import CotanComp
from omtools.core.variable import Variable


def cotan(expr):
    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")
    out = Variable()
    out.shape = expr.shape
    out.add_dependency_node(expr)

    out.build = lambda: CotanComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
        val=expr.val,
    )
    return out
