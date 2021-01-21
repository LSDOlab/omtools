from collections.abc import Iterable
from contextlib import contextmanager
from typing import Callable, Dict, Tuple

from openmdao.api import Group as OMGroup
from openmdao.core.system import System

# from omtools.core._group import _Group
from omtools.core.explicit_output import ExplicitOutput
from omtools.core.expression import Expression
from omtools.core.graph import remove_indirect_predecessors, topological_sort
from omtools.core.implicit_output import ImplicitOutput
from omtools.core.indep import Indep
from omtools.core.input import Input
from omtools.core.output import Output
from omtools.core.subsystem import Subsystem
from omtools.utils.ensure_subsystems_are_added import \
    ensure_subsystems_are_added


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
        User defined method to define expressions and add subsystems for
        model execution
        """
        # The user-defined Group.setup() method
        func(self)

        # Create a record of all nodes in DAG
        self._root.register_nodes(self.nodes)

        # Ensure that all subsystems that registerd outputs depend on
        # are considered in topological sort
        for registered_output in self._root.predecessors:
            ensure_subsystems_are_added(registered_output)
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
            if isinstance(expr, Output):
                if expr.defined == False:
                    raise ValueError("Output not defined for ", expr)

            # Construct Component object corresponding to Expression
            # object, if applicable.
            # Input objects and root Expression object do not have
            # a build method defined.
            if expr.build is not None:
                sys = expr.build(expr.name)
                pfx = 'comp_'
                promotes = ['*']
                promotes_inputs = None
                promotes_outputs = None
                if isinstance(sys, OMGroup):
                    pfx = ''
                    promotes = expr.promotes
                    promotes_inputs = expr.promotes_inputs
                    promotes_outputs = expr.promotes_outputs
                OMGroup.add_subsystem(
                    self,
                    pfx + expr.name,
                    sys,
                    promotes=promotes,
                    promotes_inputs=promotes_inputs,
                    promotes_outputs=promotes_outputs,
                )

            # Set initial values for inputs
            if isinstance(expr, Input):
                self.set_input_defaults(expr.name, val=expr.val)

            # Set design variables
            if isinstance(expr, Indep):
                if expr.dv == True:
                    self.add_design_var(expr.name)

            # Cut down on memory consumption
            del expr

    return _sort_expressions_and_build_components


class _ComponentBuilder(type):
    def __new__(cls, name, bases, attr):
        attr['setup'] = _post_setup(attr['setup'])
        return super(_ComponentBuilder, cls).__new__(cls, name, bases, attr)


class Group(OMGroup, metaclass=_ComponentBuilder):
    """
    The ``omtools.Group`` class builds ``openmdao.Component`` objects
    from Python-like expressions and adds their corresponding subsystems
    by constructing stock ``openmdao.Component`` objects.

    In ``self.setup``, first, the user declares inputs, writes
    expressions, and registers outputs. After ``self.setup`` runs,
    ``self`` builds a Directed Acyclic Graph (DAG) from registered
    outputs, analyzes the DAG to determine execution order, and adds the
    appropriate subsystems.

    In addition to supporting an expression-based style of defining a
    subsystem, ``omtools.Group`` also supports adding a subystem defined
    using a subclass of ``omtools.Group`` or ``openmdao.System``.

    The ``omtools.Group`` class only allows for expressions that define
    explicit relationships.
    For defining models that use implicit relationships and defining
    residuals, see ``omtools.ImplicitGroup``.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nodes: dict = {}
        self.input_vals: dict = {}
        self.sorted_builders = []
        self.reverse_branch_sorting: bool = False
        self._root = Expression()
        self._most_recently_added_subsystem: Subsystem = None
        self.res_out_map: Dict[str, str] = dict()
        self.brackets_map = None
        self.out_vals = dict()

    def initialize(self, *args, **kwargs):
        """
        User defined method to set options
        """
        pass

    def setup(self):
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

    def create_indep_var(
        self,
        name: str,
        shape: Tuple[int] = (1, ),
        val=1,
        # units=None,
        dv: bool = False,
    ) -> Indep:
        """
        Create a value that is constant during model evaluation

        Parameters
        ----------
        name: str
            Name of variable in OpenMDAO to be computed by
            ``ExplicitComponent`` objects connected in a cycle, or by an
            ``ExplicitComponent`` that concatenates variables
        shape: Tuple[int]
            Shape of variable
        val: Number or ndarray
            Value for variable during first model evaluation
        dv: bool
            Flag to set design variable

        Returns
        -------
        Indep
            An object to use in expressions
        """
        indep = Indep(name, shape=shape, val=val, dv=False)
        # NOTE: We choose to always include IndepVarComp objects, even
        # if they are not used by other Component objects
        self.register_output(name, indep)
        return indep

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

    def create_implicit_output(
            self,
            name: str,
            shape: Tuple[int] = (1, ),
            val=1,
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
            self,
            name,
            shape=shape,
            val=val,
        )
        # self._root.add_predecessor_node(im)
        return im

    def register_output(self, name: str,
                        expr: ExplicitOutput) -> ExplicitOutput:
        """
        Register ``expr`` as an output of the ``Group``.
        When adding subsystems, each of the subsystem's inputs requires
        a call to ``register_output`` prior to the call to
        ``add_subsystem``.

        Parameters
        ----------
        name: str
            Name of variable in OpenMDAO

        expr: Expression
            Expression that computes output

        Returns
        -------
        Expression
            Expression that computes output
        """
        if isinstance(expr, Input):
            raise TypeError("Cannot register input " + expr + " as an output")

        if expr in self._root.predecessors:
            raise ValueError(
                "Cannot register output twice; attempting to register " +
                expr.name + " as " + name)

        expr.name = name
        self._root.add_predecessor_node(expr)
        return expr

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

    @contextmanager
    def create_group(self, name: str):
        """
        Create a ``Group`` object and add as a subsystem, promoting all
        inputs and outputs.
        For use in ``with`` contexts.

        Parameters
        ----------
        name: str
            Name of new child ``Group`` object
        """
        try:
            group = Group()
            self.add_subsystem(name, group, promotes=['*'])
            yield group
        finally:
            group.setup()
            pass
