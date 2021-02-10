from omtools.comps.log10comp import Log10Comp
from omtools.core.expression import Expression


def log10(expr):
    if not isinstance(expr, Expression):
        raise TypeError(expr, " is not an Expression object")
    out = Expression()
    out.shape = expr.shape
    out.add_dependency_node(expr)
    out.build = lambda: Log10Comp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
    )
    return out
