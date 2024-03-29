from omtools.core.variable import Variable
from typing import List


def remove_indirect_dependencies(node: Variable):
    """
    Remove the dependencies that do not constrin execution order. That
    is, if C depends on B and A, and B depends on A, then the execution
    order must be A, B, C, even without the dependence of C on A.

    Parameters
    ----------
    node: Variable
        The node to treat as "root". In ``omtools.group``,
        ``Group._root`` is treated as the "root" node.
    """
    # List of indices corresponding to child node references to remove
    remove: list = []
    for child in node.dependencies:
        for grandchild in child.dependencies:
            index = node.get_dependency_index(grandchild)
            if index is not None:
                remove.append(index)
    # remove duplicate indices
    remove = list(set(remove))
    terminal_index = 0
    # children form cycle
    # TODO: explain better
    if len(remove) == len(node.dependencies):
        terminal_index = 1
    for i in reversed(remove):
        if i >= terminal_index:
            node.remove_dependency_by_index(i)


def topological_sort(node: Variable) -> List[Variable]:
    """
    Perform a topological sort on the Directed Acyclic Graph (DAG).
    If any cycles are detected when traversing the graph,
    ``topological_sort`` will not terminate, and it will cause a memory
    overflow.

    This version of a topological sort is modified so that a node will
    not be added to the sorted list until the node has been visited as
    many times as its in-degree; i.e. the number of dependents.

    Parameters
    ----------
    node: Variable
        The node to treat as "root". In ``omtools.group``,
        ``Group._root`` is treated as the "root" node.

    Returns
    -------
    list[Variable]
        List of ``Variable`` objects sorted from root to leaf. When
        overriding ``omtools.Group.setup``, the first node will be
        ``Group._root``, and the last will be an ``Input``,
        ``ExplicitOutput``, ``ImplicitOutput``, or ``IndepVar``.
    """

    # FIXME: Order of nodes is not deterministic

    # To convert between graph theory terminology and terminology used
    # to refer to expressions and their dependencies/dependents:
    # node <--> expression
    # child <--> dependency
    # parent <--> dependent

    sorted_nodes = []
    stack = [node]
    while stack != []:
        v = stack.pop()
        # Use <= instead of < to ensure that the root node (with zero
        # dependents) is visited; otherwise, no nodes will be added to
        # the list of sorted nodes
        if v.times_visited <= v.get_num_dependents():
            # Iterative Depth First Search (DFS) for a DAG, but node is
            # added to the list of sorted nodes only if all of its parents
            # have been added to the list of sorted nodes;
            # the >= condition ensures that a node with no dependents is
            # never added
            v.incr_times_visited()
            if v.times_visited >= v.get_num_dependents():
                for w in v.dependencies:
                    stack.append(w)

            if v.times_visited == v.get_num_dependents():
                sorted_nodes.append(v)
    return sorted_nodes
