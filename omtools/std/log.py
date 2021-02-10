from omtools.comps.logcomp import LogComp
from omtools.core.expression import Expression


def log(expr):
    if not isinstance(expr, Expression):
        raise TypeError(expr, " is not an Expression object")
    out = Expression()
    out.shape = expr.shape
    out.add_dependency_node(expr)
    out.build = lambda: LogComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
    )
    return out
