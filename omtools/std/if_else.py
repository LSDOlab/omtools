from omtools.core.variable import Variable
from omtools.comps.conditional_component import ConditionalComponent


def if_else(
    condition: Variable,
    expr_true: Variable,
    expr_false: Variable,
):
    if expr_true.shape != expr_false.shape:
        raise ValueError(
            "Variable shapes must be the same for Variable objects for both branches of execution"
        )

    out = Variable()
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
