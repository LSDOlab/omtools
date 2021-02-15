from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

# from omtools.comps.implicit_component import ImplicitComponent
from omtools.core.variable import Variable
from omtools.core.input import Input
from omtools.core.output import Output
from omtools.core.subsystem import Subsystem
from omtools.utils.collect_input_exprs import collect_input_exprs
from omtools.utils.gen_hex_name import gen_hex_name
from omtools.utils.get_shape_val import get_shape_val
from omtools.utils.replace_output_leaf_nodes import replace_output_leaf_nodes


def replace_input_leaf_nodes(
    node: Variable,
    leaves: Dict[str, Input],
):
    """
    Replace ``Input`` objects that depend on previous subsystems
    with ``Input`` objects that do not. This is required for defining
    graphs for residuals so that ``ImplicitComponent`` objects do
    not include subsystems.
    """
    for dependency in node.dependencies:
        if isinstance(dependency, Input):
            if len(dependency.dependencies) > 0:
                node.remove_dependency_node(dependency)
                if dependency._id in leaves.keys():
                    node.add_dependency_node(leaves[dependency._id])
                else:
                    leaf = Input(dependency.name,
                                 shape=dependency.shape,
                                 val=dependency.val)
                    leaf._id = dependency._id
                    node.add_dependency_node(leaf)
                    leaves[dependency._id] = leaf
        replace_input_leaf_nodes(dependency, leaves)


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

    def define_residual(
        self,
        residual_expr: Variable,
    ):
        """
        Define the residual that must equal zero for this output to be
        computed

        Parameters
        ----------
        residual_expr: Variable
            Residual expression
        """
        if residual_expr is self:
            raise ValueError("Variable for residual of " + self.name +
                             " cannot be self")
        if self.defined == True:
            raise ValueError("Variable for residual of " + self.name +
                             " is already defined")
        # Replace leaf nodes of residual Variable object that
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
        self.group.out_vals[self.name] = self.val

        self.defined = True

    def define_residual_bracketed(
        self,
        residual_expr: Variable,
        x1=0.,
        x2=1.,
    ):
        """
        Define the residual that must equal zero for this output to be
        computed

        Parameters
        ----------
        residual_expr: Variable
            Residual expression
        """
        if residual_expr is self:
            raise ValueError("Variable for residual of " + self.name +
                             " cannot be self")
        if self.defined == True:
            raise ValueError("Variable for residual of " + self.name +
                             " is already defined")
        # Replace leaf nodes of residual Variable object that
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
        self.group.brackets_map = (dict(), dict())
        self.group.brackets_map[0][self.name] = x1
        self.group.brackets_map[1][self.name] = x2

        self.defined = True

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        return "Implicit Output ('" + self.name + "', " + shape_str + ")"
