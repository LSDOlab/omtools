from omtools.core.expression import Expression
from omtools.core.input import Input
from typing import List


def collect_input_exprs(inputs: list, expr: Expression) -> List[Expression]:
    """
    Collect input nodes so that the resulting ``ImplicitComponent`` has
    access to inputs outside of itself.
    """
    for pred in expr.predecessors:
        if isinstance(pred, Input) == True:
            inputs.append(pred)
        inputs = collect_input_exprs(inputs, pred)
    return inputs
