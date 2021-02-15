from omtools.core.variable import Variable
from omtools.core.input import Input
from omtools.core.output import Output


def replace_output_leaf_nodes(
    root: Output,
    node: Variable,
    leaf: Input,
):
    """
    Replace ``Output`` objects that are used before they are defined
    with ``Input`` objects with same data.
    """
    for dependency in node.dependencies:
        if dependency is root:
            # replace dependency reference with Input node
            node.remove_dependency_node(dependency)
            node.add_dependency_node(leaf)
        replace_output_leaf_nodes(root, dependency, leaf)
