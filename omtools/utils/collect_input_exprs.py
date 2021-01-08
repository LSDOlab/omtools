from typing import List

from omtools.core.expression import Expression
from omtools.core.input import Input


def collect_input_exprs(
    inputs: list,
    root: Expression,
    expr: Expression,
) -> List[Expression]:
    """
    Collect input nodes so that the resulting ``ImplicitComponent`` has
    access to inputs outside of itself.
    """
    for pred in expr.predecessors:
        if pred.name != root.name:
            if isinstance(pred, Input) == True and len(pred.predecessors) == 0:
                inputs.append(pred)
            inputs = collect_input_exprs(inputs, root, pred)
    return inputs
