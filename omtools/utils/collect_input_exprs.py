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
    for dependency in expr.dependencies:
        if dependency.name != root.name:
            if isinstance(dependency, Input) == True and len(
                    dependency.dependencies) == 0:
                inputs.append(dependency)
            inputs = collect_input_exprs(inputs, root, dependency)
    return inputs
