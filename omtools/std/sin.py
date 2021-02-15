import numbers

from omtools.comps.sincomp import SinComp
from omtools.core.variable import Variable
from omtools.core.indep import Indep
from omtools.core.variable import Variable


def sin(expr):
    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")
    out = Variable()
    out.shape = expr.shape
    out.add_dependency_node(expr)
    out.build = lambda: SinComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
        val=expr.val,
    )
    return out
