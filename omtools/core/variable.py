import numbers
from typing import Optional, Tuple, Union

import numpy as np

from omtools.comps.linearcombination_comp import LinearCombinationComp
from omtools.comps.arithmetic_comps.power_combination_comp import PowerCombinationComp
from omtools.utils.gen_hex_name import gen_hex_name
from omtools.utils.slice_to_list import slice_to_list

# from omtools.std.binops import (ElementwiseAddition, ElementwiseMultiplication,
# ElementwisePower, ElementwiseSubtraction)


def slice_to_tuple(key: slice, size: int) -> tuple:
    if key.start is None:
        key = slice(0, key.stop, key.step)
    if key.stop is None:
        key = slice(key.start, size, key.step)
    return (key.start, key.stop, key.step)


class Variable():
    """
    The ``Variable`` class is a base type for nodes in a Directed
    Acyclic Graph (DAG) that represents the computation to be performed
    during model evaluation.

    Each ``Variable`` object stores a function that constructs an
    OpenMDAO ``Component`` object corresponding to the computation that
    the ``Variable`` object represents.
    """

    # A counter for all Variable objects created so far
    _count = -1

    def __pos__(self):
        return self

    def __neg__(self):
        return ElementwiseMultiplication(self, -1)

    def __add__(self, other):
        """
        Represent addition between two expressions
        """
        return ElementwiseAddition(self, other)

    def __sub__(self, other):
        """
        Represent subtraction between two expressions
        """
        return ElementwiseSubtraction(self, other)

    def __mul__(self, other):
        """
        Represent multiplication between two expressions
        """
        return ElementwiseMultiplication(self, other)

    def __truediv__(self, other):
        """
        Represent division between two expressions
        """
        return ElementwiseDivision(self, other)

    def __pow__(self, other):
        """
        Represent raising an expression to a real number
        """
        return ElementwisePower(self, other)

    def __radd__(self, other):
        """
        Represent addition between two expressions
        """
        if isinstance(other, numbers.Number) == False:
            raise TypeError(other, " is not an Variable object or literal")
        return ElementwiseAddition(other, self)

    def __rsub__(self, other):
        """
        Represent subtraction between two expressions
        """
        if isinstance(other, numbers.Number) == False:
            raise TypeError(other, " is not an Variable object or literal")
        return ElementwiseSubtraction(other, self)

    def __rmul__(self, other):
        """
        Represent multiplication between two expressions
        """
        if isinstance(other, numbers.Number) == False:
            raise TypeError(other, " is not an Variable or a literal value")
        return ElementwiseMultiplication(other, self)

    def __rtruediv__(self, other):
        """
        Represent division between two expressions
        """
        if isinstance(other, numbers.Number) == False:
            raise TypeError(other, " is not an Variable or a literal value")
        return ElementwiseDivision(other, self)

    def initialize(self, *args, **kwargs):
        for k, v in kwargs.items():
            if k == 'name':
                self.name = _id
            if k == 'shape':
                self.shape = v
            if k == 'val':
                self.val = v
            if k == 'units':
                self.units = v

    def __init__(self, *args, **kwargs):
        Variable._count += 1
        _id = gen_hex_name(Variable._count)
        self._id = _id
        self.name = _id
        self.shape = (1, )
        self.val = 1,
        self.units = None
        self.dependencies: list = []
        self.dependents: list = []
        self.build = None
        self.times_visited = 0
        self._dag_cost = 1
        self.is_residual: bool = False
        self.initialize(*args, **kwargs)
        self._getitem_called = False
        self._decomp = None
        self.indexed_exprs = dict()
        self.src_indices = dict()

    def __getitem__(
        self,
        key: Union[int, slice, Tuple[slice]],
    ):
        if self._getitem_called == False:
            self._getitem_called = True
            self._decomp = Variable(shape=self.shape, val=self.val[key])
            self._decomp.name = 'decompose_' + self.name
            self._decomp.add_dependency_node(self)

        # store key as a tuple of tuples of ints
        # no duplicate keys are stored
        # NOTE: slices are unhashable, so we can't store slices directly
        if isinstance(key, int):
            key = ((key, key + 1, None), )
        elif isinstance(key, slice):
            key = (slice_to_tuple(
                key,
                np.prod(self.shape),
            ), )
        elif isinstance(key, tuple):
            l = []
            for i in range(len(key)):
                if isinstance(key[i], slice):
                    l.append(slice_to_tuple(
                        key[i],
                        self.shape[i],
                    ))
                elif isinstance(key[i], int):
                    l.append(
                        slice_to_tuple(
                            slice(key[i], key[i] + 1, None),
                            self.shape,
                        ))
            key = tuple(l)
        else:
            raise TypeError(
                "Key must be an int, slice, or tuple object containing slice and/or int objects"
            )

        # return expression if reusing key
        if key in self._decomp.indexed_exprs.keys():
            return self._decomp.indexed_exprs[key]

        # Get flat indices from key to define corresponding component
        slices = [slice_to_list(s[0], s[1], s[2]) for s in list(key)]
        src_indices = np.ravel_multi_index(
            tuple(np.array(np.meshgrid(*slices, indexing='ij'))),
            self.shape,
        ).flatten()

        # Check size
        if np.amax(src_indices) >= np.prod(self.shape):
            raise ValueError("Indices given are out of range for " +
                             self.__repr__())

        # Create and store expression to return
        val = self.val[tuple([slice(s[0], s[1], s[2]) for s in list(key)])]
        expr = Variable(shape=val.shape, val=val)
        self._decomp.indexed_exprs[key] = expr
        expr.add_dependency_node(self._decomp)
        self._decomp.src_indices[expr] = src_indices

        # store function to construct component
        from omtools.comps.decompose_comp import DecomposeComp
        self._decomp.build = lambda: DecomposeComp(
            in_name=self.name,
            expr=self._decomp,
            val=self.val,
        )
        return expr

    def add_fwd_edges(self):
        for dependency in self.dependencies:
            dependency.add_dependent_node(self)
            dependency.add_fwd_edges()

    def add_dependent_node(self, dependent):
        self.dependents.append(dependent)
        self.dependents = list(set(self.dependents))

    def add_dependency_node(self, dependency):
        """
        Add a dependency node to the DAG representing the computation
        that OpenMDAO will perform during model evaluation. Each
        ``Variable`` object represents some computation that returns a
        value. The number of dependency nodes of an ``Variable``
        object is the number of arguments required to return a value.
        for example, in ``c = a + b``, the object ``c`` has two
        dependencies: ``a`` and ``b``.

        Parameters
        ----------
        predecesor: Variable
            An Variable object upon which this Variable object
            depends
        """
        if dependency.is_residual == True:
            raise ValueError(
                dependency.name +
                " already used as residual; cannot use in another expression")

        # Add dependency
        self.dependencies.append(dependency)
        self._dedup_dependencies()

    def register_nodes(self, nodes: dict):
        """
        Register all nodes in DAG.

        Parameters
        ----------
        nodes: dict[Variable]
            Dictionary of nodes registered so far
        """
        for node in self.dependencies:
            # Check for name collisions
            if node._id in nodes.keys():
                if nodes[node._id] is not None:
                    # if name is in use and a separate object is already
                    # registered, then we have a name collision
                    if nodes[node._id] is not node:
                        raise ValueError(
                            "Name collision (", node.name, ") between ",
                            nodes[node._id], " and ", node,
                            "; check that calls to regiser_output and ",
                            "create_* do not give the same name to ",
                            "two outputs")
            else:
                # register node
                nodes[node._id] = node
                node.register_nodes(nodes)

    def incr_times_visited(self):
        """
        Increment number of times a node is visited during ``topological_sort``.
        This is necessary for ``topological_sort`` to determine
        execution order for expressions.
        """
        self.times_visited += 1

    def get_dependency_index(self, candidate) -> Optional[int]:
        """
        Get index of dependency in ``self.dependencies``. Used for
        removing indirect dependencies that woud otherwise affect the
        cost of branches in the DAG, which would affect execution order,
        even with the sme constraints on execution order.

        Parameters
        ----------
        candidate: Variable
            The candidate dependency node

        Returns
        -------
        Optional[int]
            If ``dependency`` is a dependency of ``self``, then the index of
            ``dependency`` in ``self.dependencies`` is returned. Otherwise,
            ``None`` is returned.
        """
        for index in range(len(self.dependencies)):
            if self.dependencies[index] is candidate:
                return index
        return None

    def remove_dependency_by_index(self, index):
        """
        Remove dependency node, given its index. does nothing if
        ``index`` is out of range. See
        ``Variable.remove_dependency``.

        Parameters
        ----------
        index: int
            Index within ``self.dependencies`` where the node to be
            removed might be
        """
        if index < len(self.dependencies):
            self.dependencies.remove(self.dependencies[index])

    def compute_dag_cost(self) -> int:
        """
        Compute cost of traversing all nodes in the DAG starting at
        ``self``; ignores cycles, so it is guaranteed to compute finite
        values of cost and terminate; required for sorting branches
        before performing ``topological_sort``

        Parameters
        ----------
        branch: set[Variable]
            Set of all Variable objects traversed so far in current
            branch. If a dependency is in ``branch``, then
            ``compute_dag_cost`` will terminate and return a value. If
            ``self`` is a leaf node, then ``compute_dag_cost`` will
            terminate.

        Returns
        -------
        int
           Cost of traversing DAG starting with ``self``
        """
        for dependency in self.dependencies:
            self._dag_cost += dependency.compute_dag_cost()
        return self._dag_cost

    def sort_dependency_branches(self, reverse_branch_sorting=False):
        """
        Sort dependencies by DAG cost so that branches with higher cost
        (``Variable._dag_cost``) appear before shorter branches
        ("critical path" sorting). User can set flag to force branches
        with lower cost to appear before branches with hgiher cost ("low
        hanging fruit" sorting).

        Parameters
        ----------
        reverse_branch_sorting: bool = False
            Flag to set sorting preference. Default is "critical path".
            Reverse branch sorting to produce "low hanging fruit"
            sorting.
        """
        self.dependencies = sorted(
            self.dependencies,
            key=lambda x: x._dag_cost,
            reverse=reverse_branch_sorting,
        )

    def _dedup_dependencies(self):
        """
        Remove duplicate dependencies. Used when adding a dependency.
        """
        self.dependencies = list(set(self.dependencies))

    def remove_dependency_node(self, candidate):
        """
        Remove dependency node. Does nothing if `candidate` is not a
        dependency. Used for removing indirect dependencies and
        preventing cycles from forming in DAG.

        Parameters
        ----------
        candidate: Variable
            Node to remove from ``self.dependencies``
        """
        index = self.get_dependency_index(candidate)
        if index is not None:
            self.remove_dependency_by_index(index)

    def print_dag(self, depth=-1, indent=''):
        """
        Print the graph starting at this node (debugging tool)
        """
        print(indent, id(self), self.name, len(self.dependents),
              self.times_visited, self)
        if len(self.dependencies) == 0:
            print(self.name, 'has no dependencies')
        if depth > 0:
            depth -= 1
        if depth != 0:
            for dependency in self.dependencies:
                dependency.print_dag(depth=depth, indent=indent + ' ')

    def get_num_dependents(self):
        return len(self.dependents)


class ElementwiseAddition(Variable):
    """
    ``Variable`` object used to create a LinearCombinationComp for
    addition
    """
    def initialize(self, expr1, expr2):
        if isinstance(expr1, Variable):
            self.shape = expr1.shape
            self.add_dependency_node(expr1)
        if isinstance(expr2, Variable):
            self.shape = expr2.shape
            self.add_dependency_node(expr2)
        if isinstance(expr1, Variable) and isinstance(expr2, Variable):
            if expr1.shape == expr2.shape:
                self.shape = expr1.shape

                self.build = lambda: LinearCombinationComp(
                    shape=expr1.shape,
                    out_name=self.name,
                    in_names=[expr1.name, expr2.name],
                    coeffs=1,
                    in_vals=[expr1.val, expr2.val],
                )

            else:
                raise ValueError(
                    "Shapes for expressions " + repr(expr1) + " and " +
                    repr(expr2) +
                    " do not match. If shapes are as intended (e.g. multiply scalar times array), use `omtools.expand` expression to mimick broadcasting."
                )

        if (isinstance(expr1, numbers.Number) or isinstance(
                expr1, np.ndarray)) and isinstance(expr2, Variable):

            self.build = lambda: LinearCombinationComp(
                shape=expr2.shape,
                out_name=self.name,
                in_names=[expr2.name],
                coeffs=1,
                constant=expr1,
                in_vals=[expr2.val],
            )

        if (isinstance(expr2, numbers.Number) or isinstance(
                expr2, np.ndarray)) and isinstance(expr1, Variable):

            self.build = lambda: LinearCombinationComp(
                shape=expr1.shape,
                out_name=self.name,
                in_names=[expr1.name],
                coeffs=1,
                constant=expr2,
                in_vals=[expr1.val],
            )

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
            shape_str += ")"
        return "ElementwiseAddition (" + shape_str + ")"


class ElementwiseSubtraction(Variable):
    """
    ``Variable`` object used to create a LinearCombinationComp for
    subtraction
    """
    def initialize(self, expr1, expr2):
        if isinstance(expr1, Variable):
            self.shape = expr1.shape
            self.add_dependency_node(expr1)
        if isinstance(expr2, Variable):
            self.shape = expr2.shape
            self.add_dependency_node(expr2)
        if isinstance(expr1, Variable) and isinstance(expr2, Variable):
            if expr1.shape == expr2.shape:
                self.shape = expr1.shape

                self.build = lambda: LinearCombinationComp(
                    shape=expr1.shape,
                    out_name=self.name,
                    in_names=[expr1.name, expr2.name],
                    coeffs=[1, -1],
                    in_vals=[expr1.val, expr2.val],
                )

            else:
                raise ValueError(
                    "Shapes for expressions " + repr(expr1) + " and " +
                    repr(expr2) +
                    " do not match. If shapes are as intended (e.g. multiply scalar times array), use `omtools.expand` expression to mimick broadcasting."
                )

        if (isinstance(expr1, numbers.Number) or isinstance(
                expr1, np.ndarray)) and isinstance(expr2, Variable):

            self.build = lambda: LinearCombinationComp(
                shape=expr2.shape,
                out_name=self.name,
                in_names=[expr2.name],
                coeffs=-1,
                constant=expr1,
                in_vals=[expr2.val],
            )

        if (isinstance(expr2, numbers.Number) or isinstance(
                expr2, np.ndarray)) and isinstance(expr1, Variable):

            self.build = lambda: LinearCombinationComp(
                shape=expr1.shape,
                out_name=self.name,
                in_names=[expr1.name],
                coeffs=1,
                constant=-expr2,
                in_vals=[expr1.val],
            )

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
            shape_str += ")"
        return "ElementwiseSubtraction (" + shape_str + ")"


class ElementwiseMultiplication(Variable):
    """
    ``Variable`` object used to create a PowerCombinationComp for
    multiplication
    """
    def initialize(self, expr1, expr2):
        if isinstance(expr1, Variable):
            self.shape = expr1.shape
            self.add_dependency_node(expr1)
        if isinstance(expr2, Variable):
            self.shape = expr2.shape
            self.add_dependency_node(expr2)
        if isinstance(expr1, Variable) and isinstance(expr2, Variable):
            if expr1.shape == expr2.shape:
                self.shape = expr1.shape

                self.build = lambda: PowerCombinationComp(
                    shape=expr1.shape,
                    out_name=self.name,
                    in_names=[expr1.name, expr2.name],
                    powers=1,
                    in_vals=[expr1.val, expr2.val],
                )

            else:
                raise ValueError(
                    "Shapes for expressions " + repr(expr1) + " and " +
                    repr(expr2) +
                    " do not match. If shapes are as intended (e.g. multiply scalar times array), use `omtools.expand` expression to mimick broadcasting."
                )

        if (isinstance(expr1, numbers.Number) or isinstance(
                expr1, np.ndarray)) and isinstance(expr2, Variable):

            self.shape = expr2.shape

            self.build = lambda: PowerCombinationComp(
                shape=expr2.shape,
                out_name=self.name,
                in_names=[expr2.name],
                coeff=expr1,
                powers=1,
                in_vals=[expr2.val],
            )

        if (isinstance(expr2, numbers.Number) or isinstance(
                expr2, np.ndarray)) and isinstance(expr1, Variable):

            self.shape = expr1.shape

            self.build = lambda: PowerCombinationComp(
                shape=expr1.shape,
                out_name=self.name,
                in_names=[expr1.name],
                coeff=expr2,
                powers=1,
                in_vals=[expr1.val],
            )

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
            shape_str += ")"
        return "ElementwiseMultiplication (" + shape_str + ")"


class ElementwiseDivision(Variable):
    """
    ``Variable`` object used to create a PowerCombinationComp for
    multiplication
    """
    def initialize(self, expr1, expr2):
        if isinstance(expr1, Variable):
            self.shape = expr1.shape
            self.add_dependency_node(expr1)
        if isinstance(expr2, Variable):
            self.shape = expr2.shape
            self.add_dependency_node(expr2)
        if isinstance(expr1, Variable) and isinstance(expr2, Variable):
            if expr1.shape == expr2.shape:
                self.shape = expr1.shape

                self.build = lambda: PowerCombinationComp(
                    shape=expr1.shape,
                    out_name=self.name,
                    in_names=[expr1.name, expr2.name],
                    powers=[1, -1],
                    in_vals=[expr1.val, expr2.val],
                )

            else:
                raise ValueError(
                    "Shapes for expressions " + repr(expr1) + " and " +
                    repr(expr2) +
                    " do not match. If shapes are as intended (e.g. multiply scalar times array), use `omtools.expand` expression to mimick broadcasting."
                )

        if (isinstance(expr1, numbers.Number) or isinstance(
                expr1, np.ndarray)) and isinstance(expr2, Variable):

            self.shape = expr2.shape

            self.build = lambda: PowerCombinationComp(
                shape=expr2.shape,
                out_name=self.name,
                in_names=[expr2.name],
                coeff=expr1,
                powers=-1,
                in_vals=[expr2.val],
            )

        if (isinstance(expr2, numbers.Number) or isinstance(
                expr2, np.ndarray)) and isinstance(expr1, Variable):
            if expr2 == 0:
                raise ValueError("Cannot divide by zero")

            self.shape = expr1.shape

            self.build = lambda: PowerCombinationComp(
                shape=expr1.shape,
                out_name=self.name,
                in_names=[expr1.name],
                coeff=1 / expr2,
                powers=1,
                in_vals=[expr1.val],
            )

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
            shape_str += ")"
        return "ElementwiseDivision (" + shape_str + ")"


class ElementwisePower(Variable):
    """
    ``Variable`` object used to create a PowerCombinationComp for
    scalar exponent
    """
    def initialize(self, expr1, expr2):
        if isinstance(expr1, Variable):
            self.add_dependency_node(expr1)
        else:
            raise TypeError(expr1, " is not an Variable object or literal")
        if isinstance(expr2, numbers.Number):
            self.shape = expr1.shape

            self.build = lambda: PowerCombinationComp(
                shape=expr1.shape,
                out_name=self.name,
                in_names=[expr1.name],
                powers=expr2,
                in_vals=[expr1.val],
            )
        if isinstance(expr2, np.ndarray):
            if expr1.shape != expr2.shape:
                raise ValueError("Shape mismatch between base and power array")
            self.shape = expr1.shape

            self.build = lambda: PowerCombinationComp(
                shape=expr1.shape,
                out_name=self.name,
                in_names=[expr1.name],
                powers=expr2,
                in_vals=[expr1.val],
            )

        else:
            # TODO: constant array exponent
            # TODO: stock component where exponent is not a constant
            return NotImplemented

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
            shape_str += ")"
        return "ElementwisePower (" + shape_str + ")"
