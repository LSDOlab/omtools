from omtools.core.subsystem import Subsystem


def ensure_subsystems_are_added(node):
    """
    Perform a DFS and update number of successors for each subsystem
    that is a dependency of ``node``

    Parameters
    ----------
    node: Expression
        The node to treat as "root". In ``omtools.group``,
        ``Group._root`` is treated as the "root" node.

    """

    # To convert between graph theory terminology and terminology used
    # to refer to expressions and their dependencies/dependents:
    # node <--> expression
    # child <--> predecessor
    # parent <--> successor

    stack = [node]
    while stack != []:
        v = stack.pop()
        # Use <= instead of < to ensure that the root node (with zero
        # successors) is visited; otherwise, no nodes will be added to
        # the list of sorted nodes
        if v.times_visited <= v.num_successors:
            # Iterative Depth First Search (DFS) for a DAG, but node is
            # considered visited/discovered only if all of its parents
            # have been visited/discovered; the >= condition ensures
            # that a node with no successors is never added
            v.incr_times_visited()
            if v.times_visited >= v.num_successors:
                for w in v.predecessors:
                    if isinstance(w, Subsystem):
                        if w.num_inputs > 0:
                            w.incr_num_successors()
                            w.num_inputs -= 1
                    stack.append(w)
