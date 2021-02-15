from typing import List

from omtools.core.variable import Variable
from omtools.core.input import Input


def collect_input_exprs(
    inputs: list,
    root: Variable,
    expr: Variable,
) -> List[Variable]:
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
