from omtools.core.expression import Expression
from omtools.comps.conditional_component import ConditionalComponent


def if_else(
    condition: Expression,
    expr_true: Expression,
    expr_false: Expression,
):
    if expr_true.shape != expr_false.shape:
        raise ValueError(
            "Expression shapes must be the same for Expression objects for both branches of execution"
        )

    out = Expression()
    out.add_dependency_node(condition)
    out.add_dependency_node(expr_true)
    out.add_dependency_node(expr_false)
    out.build = lambda: ConditionalComponent(
        out_name=out.name,
        condition=condition,
        expr_true=expr_true,
        expr_false=expr_false,
    )
    return out
