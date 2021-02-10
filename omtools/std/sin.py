import numbers

from omtools.comps.sincomp import SinComp
from omtools.core.expression import Expression
from omtools.core.indep import Indep
from omtools.core.expression import Expression


def sin(expr):
    if not isinstance(expr, Expression):
        raise TypeError(expr, " is not an Expression object")
    out = Expression()
    out.shape = expr.shape
    out.add_dependency_node(expr)
    out.build = lambda: SinComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
    )
    return out
