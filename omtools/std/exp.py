from omtools.comps.expcomp import ExpComp
from omtools.core.expression import Expression


def exp(expr):
    if not isinstance(expr, Expression):
        raise TypeError(expr, " is not an Expression object")
    out = Expression()
    out.shape = expr.shape
    out.add_dependency_node(expr)
    out.build = lambda: ExpComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
    )
    return out
