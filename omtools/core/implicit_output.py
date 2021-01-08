from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
from openmdao.api import DirectSolver, NewtonSolver
from openmdao.solvers.solver import LinearSolver, NonlinearSolver

# from omtools.comps.implicit_component import ImplicitComponent
from omtools.core.expression import Expression
from omtools.core.input import Input
from omtools.core.output import Output
from omtools.core.subsystem import Subsystem
from omtools.utils.collect_input_exprs import collect_input_exprs
from omtools.utils.gen_hex_name import gen_hex_name
from omtools.utils.get_shape_val import get_shape_val
from omtools.utils.replace_output_leaf_nodes import replace_output_leaf_nodes


def replace_input_leaf_nodes(
    node: Expression,
    leaves: Dict[str, Input],
):
    """
    Replace ``Input`` objects that depend on previous subsystems
    with ``Input`` objects that do not. This is required for defining
    graphs for residuals so that ``ImplicitComponent`` objects do
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
        group,
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
        self.group = group
        self.name = name
        self.shape, self.val = get_shape_val(shape, val)
        self.defined = False
        self.linear_solver = DirectSolver()
        self.nonlinear_solver = NewtonSolver(solve_subsystems=False)

    def define_residual(
        self,
        residual_expr: Expression,
        linear_solver: LinearSolver = None,
        nonlinear_solver: NonlinearSolver = None,
        n2: bool = False,
    ):
        """
        Define the residual that must equal zero for this output to be
        computed

        Parameters
        ----------
        residual_expr: Expression
            Residual expression
        """
        # Replace leaf nodes of residual Expression object that
        # correspond to this ImplicitOutput node with Input objects;
        replace_output_leaf_nodes(
            self,
            residual_expr,
            Input(self.name, shape=self.shape, val=self.val),
        )

        # register expression that computes residual
        self.group.register_output(
            residual_expr.name,
            residual_expr,
        )

        # map residual name to user defined output name
        self.group.res_out_map[residual_expr.name] = self.name
        # self.group.res_brackets_map[residual_expr.name] = self.name

        # TODO: move solver assignment to component
        # # Assign solvers and update costs to reflect iterative
        # # computations
        # if linear_solver is not None:
        #     self.linear_solver = linear_solver
        #     if 'maxiter' in self.linear_solver.options._dict.keys():
        #         self._dag_cost += self.linear_solver.options['maxiter']
        # if nonlinear_solver is not None:
        #     self.nonlinear_solver = nonlinear_solver
        #     if 'maxiter' in self.nonlinear_solver.options._dict.keys():
        #         self._dag_cost += self.nonlinear_solver.options['maxiter']

    def define_residual_bracketed(
        self,
        residual_expr: Expression,
        x1=0.,
        x2=1.,
        n2: bool = False,
    ):
        """
        Define the residual that must equal zero for this output to be
        computed

        Parameters
        ----------
        residual_expr: Expression
            Residual expression
        """
        # Replace leaf nodes of residual Expression object that
        # correspond to this ImplicitOutput node with Input objects;
        replace_output_leaf_nodes(
            self,
            residual_expr,
        )

        # register expression that computes residual
        self.group.register_output(
            residual_expr.name,
            residual_expr,
        )

        # map residual name to user defined output name
        self.group.res_out_map[residual_expr.name] = self.name
        self.group.brackets_map = (dict(), dict())
        self.group.brackets_map[0][self.name] = x1
        self.group.brackets_map[1][self.name] = x2

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        return "Implicit Output ('" + self.name + "', " + shape_str + ")"
