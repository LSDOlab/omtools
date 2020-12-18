from omtools.core.expression import Expression
from omtools.core.input import Input
from omtools.core.output import Output


def replace_output_leaf_nodes(
    root: Output,
    node: Expression,
    leaf: Input,
):
    """
    Replace ``Output`` objects that are used before they are defined
    with ``Input`` objects with same data.
    """
    for pred in node.predecessors:
        if pred is root:
            # replace predecessor reference with Input node
            node.remove_predecessor_node(pred)
            node.add_predecessor_node(leaf)
        replace_output_leaf_nodes(root, pred, leaf)
