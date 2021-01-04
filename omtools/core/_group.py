from typing import Callable, Dict, Tuple

from openmdao.api import Group as OMGroup
from openmdao.core.system import System

from omtools.core.output import Output
from omtools.core.explicit_output import ExplicitOutput
from omtools.core.expression import Expression
from omtools.core.graph import remove_indirect_predecessors, topological_sort
from omtools.core.implicit_output import ImplicitOutput
from omtools.core.indep import Indep
from omtools.core.input import Input
from omtools.core.subsystem import Subsystem
from omtools.utils.ensure_subsystems_are_added import \
    ensure_subsystems_are_added
from collections.abc import Iterable


class _Group(OMGroup):
    """
    The base class for the ``omtools.Group`` and
    ``omtools.ImplicitGroup`` classes.
    Users extend the `omtools.Group`` and ``omtools.ImplicitGroup``
    classes, which build ``openmdao.Component`` objects
    from Python-like expressions and add their corresponding subsystems
    by constructing stock ``openmdao.Component`` objects provided by
    ``omtools``.

    In ``self.setup``, first, the user declares inputs, writes
    expressions, and registers outputs. After ``self.setup`` runs,
    ``self`` builds a Directed Acyclic Graph (DAG) from registered
    outputs, analyzes the DAG to determine execution order, and adds the
    appropriate subsystems.

    In addition to supporting an expression-based style of defining a
    subsystem, ``omtools.Group`` also supports adding a subystem defined
    using a subclass of ``omtools.Group`` or ``openmdao.System``.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nodes: dict = {}
        self.input_vals: dict = {}
        self.sorted_builders = []
        self.reverse_branch_sorting: bool = False
        self._root = Expression()
        self._most_recently_added_subsystem: Subsystem = None

    def initialize(self, *args, **kwargs):
        """
        User defined method to set options
        """
        pass

    def setup(self):
        """
        User defined method to define expressions and add subsystems for
        model execution
        """
        pass

    def declare_input(
            self,
            name: str,
            shape: Tuple[int] = (1, ),
            val=1,
        # units=None,
    ) -> Input:
        """
        Declare an input to use in an expression.

        An input can be an output of a child ``System``. If the user
        declares an input that is computed by a child ``System``, then
        the call to ``self.declare_input`` must appear after the call to
        ``self.add_subsystem``.

        Parameters
        ----------
        name: str
            Name of variable in OpenMDAO to be used as an input in
            generated ``Component`` objects
        shape: Tuple[int]
            Shape of variable
        val: Number or ndarray
            Default value for variable

        Returns
        -------
        Input
            An object to use in expressions
        """
        inp = Input(
            name,
            shape=shape,
            val=val,
            # units=units,
        )
        if self._most_recently_added_subsystem is not None:
            inp.add_predecessor_node(self._most_recently_added_subsystem)
            # This is to guarantee that the subsystem is added even if
            # outputs that depend on the subsystem are not registered
            self._most_recently_added_subsystem.decr_num_successors()
            self._most_recently_added_subsystem.num_inputs += 1
        return inp

    def create_output(
            self,
            name: str,
            shape: Tuple[int] = (1, ),
            val=1,
    ) -> ExplicitOutput:
        """
        Create a value that is computed explicitly

        Parameters
        ----------
        name: str
            Name of variable in OpenMDAO to be computed by
            ``ExplicitComponent`` objects connected in a cycle, or by an
            ``ExplicitComponent`` that concatenates variables
        shape: Tuple[int]
            Shape of variable

        Returns
        -------
        ExplicitOutput
            An object to use in expressions
        """
        ex = ExplicitOutput(
            name,
            shape=shape,
            val=val,
        )
        self._root.add_predecessor_node(ex)
        return ex

    def add_subsystem(
        self,
        name: str,
        subsys: System,
        promotes: Iterable = None,
        promotes_inputs: Iterable = None,
        promotes_outputs: Iterable = None,
    ):
        """
        Add a subsystem to the ``Group``.

        ``self.add_subsystem`` call must be preceded by a call to
        ``self.register_output`` for each of the subsystem's inputs,
        and followed by ``self.declare_input`` for each of the
        subsystem's outputs.

        Parameters
        ----------
        name: str
            Name of subsystem
        subsys: System
            Subsystem to add to `Group`
        promotes: Iterable
            Variables to promote
        promotes_inputs: Iterable
            Inputs to promote
        promotes_outputs: Iterable
            Outputs to promote

        Returns
        -------
        System
            Subsystem to add to `Group`
        """
        self._most_recently_added_subsystem = Subsystem(
            name,
            subsys,
            promotes,
            promotes_inputs=promotes_inputs,
            promotes_outputs=promotes_outputs,
        )
        self._most_recently_added_subsystem.add_predecessor_node(self._root)
        new_root = Expression()
        new_root.add_predecessor_node(self._most_recently_added_subsystem)
        self._root = new_root
        return subsys
