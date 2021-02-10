from omtools.comps.coscomp import CosComp
from omtools.core.expression import Expression


def cos(expr):
    if not isinstance(expr, Expression):
        raise TypeError(expr, " is not an Expression object")
    out = Expression()
    out.shape = expr.shape
    out.add_dependency_node(expr)

    out.build = lambda: CosComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
    )
    return out
