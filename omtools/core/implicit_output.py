from omtools.core.expression import Expression
from omtools.core.input import Input
from omtools.core.output import Output
from omtools.core.subsystem import Subsystem
from omtools.utils.collect_input_exprs import collect_input_exprs
from omtools.utils.comps.composite_implicit_comp import CompositeImplicitComp
from omtools.utils.replace_output_leaf_nodes import replace_output_leaf_nodes
from omtools.utils.gen_hex_name import gen_hex_name
from copy import deepcopy
from typing import Dict, Tuple
from openmdao.api import DirectSolver, NewtonSolver
from openmdao.solvers.solver import LinearSolver, NonlinearSolver
from omtools.utils.get_shape import get_shape


def replace_input_leaf_nodes(
    node: Expression,
    leaves: Dict[str, Input],
):
    """
    Replace ``Input`` objects that depend on previous subsystems
    with ``Input`` objects that do not. This is required for defining
    graphs for residuals so that ``CompositeImplicitComp`` objects do
    not include subsystems.
    """
    for pred in node.predecessors:
        if isinstance(pred, Input):
            if len(pred.predecessors) > 0:
                node.remove_predecessor_node(pred)
                if pred._id in leaves.keys():
                    node.add_predecessor_node(leaves[pred._id])
                else:
                    leaf = Input(pred.name, shape=pred.shape, val=pred.val)
                    leaf._id = pred._id
                    node.add_predecessor_node(leaf)
                    leaves[pred._id] = leaf
        replace_input_leaf_nodes(pred, leaves)


class ImplicitOutput(Output):
    """
    Class for creating an implicit output
    """
    def initialize(
        self,
        name: str,
        shape: Tuple[int] = (1, ),
        val=1,
        linear_solver: LinearSolver = None,
        nonlinear_solver: NonlinearSolver = None,
    ):
        """
        Initialize implicit output

        Parameters
        ----------
        name: str
            Name of variable to compute implicitly
        shape: Tuple[int]
            Shape of variable to compute implicitly
        val: Number or ndarray
            Initial value of variable to compute implicitly
        """
        self.name = name
        self.shape = get_shape(shape, val)
        self.val = val
        self.defined = False
        self.linear_solver = DirectSolver()
        self.nonlinear_solver = NewtonSolver(solve_subsystems=False)

    def define_residual(
        self,
        residual_expr: Expression,
        linear_solver: LinearSolver = None,
        nonlinear_solver: NonlinearSolver = None,
    ):
        """
        Define the residual that must equal zero for this output to be
        computed

        Parameters
        ----------
        residual_expr: Expression
            Residual expression
        """
        if residual_expr is self:
            raise ValueError("Expression for residual of " + self.name +
                             " cannot be self")
        if self.defined == True:
            raise ValueError("Expression for residual of " + self.name +
                             " is already defined")

        # Establish direct dependence of ImplicitOutput object on Input
        # objects, which depend on most recently added subsystem
        input_exprs = collect_input_exprs([], residual_expr)
        for input_expr in input_exprs:
            self.add_predecessor_node(input_expr)
            input_expr.decr_num_successors()

        # Replace leaf nodes of residual Expression object that
        # correspond to this ImplicitOutput node with Input objects;
        # cannot be called before collect_input_exprs
        replace_output_leaf_nodes(
            self,
            residual_expr,
            Input(self.name, shape=self.shape, val=self.val),
        )

        # The ImplicitOutput object directs OpenMDAO to construct a
        # CompositeImplicitComp object. The CompositeImplicitComp class
        # defines a Problem with a model that computes the residual. The
        # model contained within the
        # CompositeImplicitComp object does not contain any subsystems
        # added prior to the declared inputs for this ImplicitOutput
        # object.
        # Here, we replace the leaf nodes of residual Expression objects with Input objects that
        # do not depend on the most recently added subsystem.
        residual_expr_copy = deepcopy(residual_expr)
        replace_input_leaf_nodes(
            residual_expr_copy,
            dict(),
        )

        # Assign solvers and update costs to reflect iterative
        # computations
        if linear_solver is not None:
            self.linear_solver = linear_solver
            if 'maxiter' in self.linear_solver.options._dict.keys():
                self._dag_cost += self.linear_solver.options['maxiter']
        if nonlinear_solver is not None:
            self.nonlinear_solver = nonlinear_solver
            if 'maxiter' in self.nonlinear_solver.options._dict.keys():
                self._dag_cost += self.nonlinear_solver.options['maxiter']

        in_exprs = []
        all_in_exprs = collect_input_exprs([], residual_expr_copy)
        for expr in all_in_exprs:
            if expr.name != self.name:
                in_exprs.append(expr)

        def build(name: str):
            comp = CompositeImplicitComp(
                in_exprs=in_exprs,
                out_expr=self,
                res_expr=residual_expr_copy,
            )
            comp.linear_solver = self.linear_solver
            comp.nonlinear_solver = self.nonlinear_solver
            return comp

        self.build = build
        self.defined = True

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        return "Implicit Output ('" + self.name + "', " + shape_str + ")"


if __name__ == "__main__":
    from openmdao.api import Problem
    import omtools.api as ot

    class G(ot.Group):
        def setup(self):
            z = self.declare_input('z')
            x = self.create_implicit_output('x')
            y = 4 - (x + 0.001)**2 + z
            x.define_residual(y)

    prob = Problem()
    prob.model = G()
    prob.setup()
    prob.run_model()
