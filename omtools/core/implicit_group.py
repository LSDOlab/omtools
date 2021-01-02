from typing import Callable, Dict, Tuple

from openmdao.api import Group as OMGroup
from openmdao.core.system import System

from omtools.core._group import _Group
from omtools.core.output import Output
from omtools.core.explicit_output import ExplicitOutput
from omtools.core.implicit_output import ImplicitOutput
from omtools.core.graph import remove_indirect_predecessors, topological_sort
from omtools.core.expression import Expression
from omtools.core.indep import Indep
from omtools.core.input import Input
from omtools.core.subsystem import Subsystem
from omtools.utils.ensure_subsystems_are_added import \
    ensure_subsystems_are_added
from collections.abc import Iterable


def _post_setup(func: Callable) -> Callable:
    """
    This function replaces ``Group.setup`` with a new method that calls
    ``Group.setup`` and performs the necessary steps to determine
    execution order and construct and add the appropriate subsystems.

    The new method is the core of the ``omtools`` package. This function
    analyzes the Directed Acyclic Graph (DAG), sorts expressions, and
    directs OpenMDAO to add the corresponding ``Component`` objects.

    This function ensures an execution order that is free of unnecessary
    feedback regardless of the order in which the user registers
    outputs.
    """
    def _sort_expressions_and_build_components(self):
        """
        User defined method to define residual expressions.
        Residuals may be defined in ``omtools.ImplicitGroup`` the same
        way expressions are defined in a ``omtools.Group``.

        For each call to ``create_implicit_output``, a corresponding
        ``openmdao.ImplicitComponent`` will be created.
        No other subsystems will be added to the ``ImplicitGroup``.

        Note that all inputs to an ``ImplicitGroup`` MUST be declared at
        the top of the user-defined ``setup`` method, before any call to
        ``add_subsystem`` is made.
        """
        # The user-defined Group.setup() method
        func(self)

        # Create a record of all nodes in DAG
        self._root.register_nodes(self.nodes)

        # Ensure that all subsystems that registerd outputs depend on
        # are considered in topological sort
        for registered_output in self._root.predecessors:
            ensure_subsystems_are_added(registered_output)

        # Remove predecessors that are not ImplicitOutput objects so
        # that all subsystems added to ImplicitGroup form part of the
        # residual.
        for registered_output in self._root.predecessors:
            if isinstance(registered_output, ImplicitOutput) == False:
                self._root.remove_predecessor_node(registered_output)
        for node in self.nodes.values():
            node.times_visited = 0

        # Clean up graph, removing dependencies that do not constrain
        # execution order
        for node in self.nodes.values():
            remove_indirect_predecessors(node)

        # Compute branch costs and sort branches to get desired sparsity
        # pattern in system jacobian
        self._root.compute_dag_cost()
        for node in self.nodes.values():
            node.sort_predecessor_branches(
                reverse_branch_sorting=self.reverse_branch_sorting)

        # Sort expressions, preventing unnecessary feedbacks (i.e.
        # feedbacks will only occur if there is coupling between
        # components)
        self.sorted_expressions = topological_sort(self._root)

        # Now that expressions are sorted, construct components
        for expr in reversed(self.sorted_expressions):
            # Check if outputs are defined
            if isinstance(expr, ImplicitOutput):
                if expr.defined == False:
                    raise ValueError("Output not defined for ", expr)

                # Construct Component object corresponding to Expression
                # object, if applicable.
                # Input objects and root Expression object do not have
                # a build method defined.
                OMGroup.add_subsystem(
                    self,
                    'comp_' + expr.name,
                    expr.build(expr.name),
                    promotes=['*'],
                )

            # Set initial values for inputs
            if isinstance(expr, Input):
                self.set_input_defaults(expr.name, val=expr.val)

            # Cut down on memory consumption
            del expr

    return _sort_expressions_and_build_components


class _ComponentBuilder(type):
    def __new__(cls, name, bases, attr):
        attr['setup'] = _post_setup(attr['setup'])
        return super(_ComponentBuilder, cls).__new__(cls, name, bases, attr)


class ImplicitGroup(_Group, metaclass=_ComponentBuilder):
    """
    The ``omtools.ImplicitGroup`` users with the ability to define
    residuals that may depend on additional subsystems.

    The ``omtools.ImplicitGroup`` class only allows for expressions to
    be used as residuals, not outputs of the ``ImplicitGroup``.
    For defining models using other expressions and explicit
    relationships, see ``omtools.Group``.
    """
    def setup(self):
        pass

    def create_implicit_output(
            self,
            name: str,
            shape: Tuple[int] = (1, ),
        # val=1,
    ) -> ImplicitOutput:
        """
        Create a value that is computed implicitly

        Parameters
        ----------
        name: str
            Name of variable in OpenMDAO to be computed by an
            ``ImplicitComponent``
        shape: Tuple[int]
            Shape of variable

        Returns
        -------
        ImplicitOutput
            An object to use in expressions
        """
        im = ImplicitOutput(
            name,
            shape=shape,
            # val=val,
        )
        self._root.add_predecessor_node(im)
        return im
