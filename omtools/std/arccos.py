from omtools.comps.arccoscomp import ArccosComp
from omtools.core.expression import Expression


def arccos(expr):
    if not isinstance(expr, Expression):
        raise TypeError(expr, " is not an Expression object")
    out = Expression()
    out.shape = expr.shape
    out.add_dependency_node(expr)

    out.build = lambda: ArccosComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
    )
    return out
