from typing import Dict, List, Tuple, Union

import numpy as np

from omtools.core.expression import Expression
from omtools.core.input import Input
from omtools.core.output import Output
from omtools.comps.indexed_pass_through_comp import \
    IndexedPassThroughComp
from omtools.comps.pass_through_comp import PassThroughComp
from omtools.utils.get_shape_val import get_shape_val
from omtools.utils.replace_output_leaf_nodes import replace_output_leaf_nodes
from omtools.utils.slice_to_list import slice_to_list


class ExplicitOutput(Output):
    """
    Class for creating an explicit output
    """
    def initialize(
            self,
            name: str,
            shape: Tuple[int] = (1, ),
            val=1,
    ):
        """
        Initialize explicit output

        Parameters
        ----------
        name: str
            Name of variable to compute explicitly
        shape: Tuple[int]
            Shape of variable to compute explicitly
        val: Number or ndarray
            Initial value of variable to compute explicitly
        """
        self.name = name
        self.shape, self.val = get_shape_val(shape, val)
        self.defined = False
        self.indexed_assignment = False
        self._tgt_indices: Dict[str, List[int]] = {}
        self.checked_indices = set()
        self.overlapping_indices = set()
        self._tgt_vals = dict()

    def define(self, expr: Expression):
        """
        Define expression (in terms of ``self``) that computes value for
        this output. This method defines a cyclic relationship, which
        requires an iterative solver to converge using OpenMDAO.

        Parameters
        ----------
        expr: Expression
            The expression to compute iteratively until convergence
        """
        if expr is self:
            raise ValueError("Expression for output " + self.name +
                             " cannot be self")
        if self.indexed_assignment == True and self.defined == True:
            raise ValueError(
                "Expression for output " + self.name +
                " is already defined using indexed assignment; use index assignment to concatenate expression outputs"
            )

        if self.defined == True:
            raise ValueError(
                "Expression for output " + self.name +
                ", which forms a cycle to be computed iteratively, is already defined"
            )
        self.defined = True

        self.add_dependency_node(expr)
        replace_output_leaf_nodes(
            self,
            self,
            Input(self.name, shape=self.shape, val=self.val),
        )

        self.build = lambda: PassThroughComp(
            expr=expr,
            name=self.name,
        )

    # TODO: index by tuple, not expression?
    # TODO: allow negative indices
    def __setitem__(
        self,
        key: Union[int, slice, Tuple[slice]],
        expr: Expression,
    ):
        self.add_dependency_node(expr)
        tgt_indices = []
        # n-d array assignment
        if isinstance(key, tuple):
            slices = [
                slice_to_list(
                    s.start,
                    s.stop,
                    s.step,
                ) for s in list(key)
            ]
            tgt_indices = np.ravel_multi_index(
                tuple(np.array(np.meshgrid(*slices, indexing='ij'))),
                self.shape,
            ).flatten()
        # 1-d array assignment
        elif isinstance(key, slice):
            tgt_indices = slice_to_list(
                key.start,
                key.stop,
                key.step,
            )
        # integer index assignment
        elif isinstance(key, int):
            tgt_indices = [key]
        else:
            raise TypeError(
                "When assigning indices of an expression, key must be an int, a slice, or a tuple of slices"
            )

        # Check size
        if np.amax(tgt_indices) >= np.prod(self.shape):
            raise ValueError("Indices given are out of range for " +
                             self.__repr__())

        if expr.name in self._tgt_indices.keys():
            raise KeyError("Repeated use of expression " + expr.name +
                           " in assignment to elements in " + self.name +
                           ". Consider using omtools.expand")
        self._tgt_indices[expr.name] = (expr.shape, tgt_indices)
        self._tgt_vals[expr.name] = expr.val

        # Check for overlapping indices
        self.overlapping_indices = self.checked_indices.intersection(
            tgt_indices)
        self.checked_indices = self.checked_indices.union(tgt_indices)
        if len(self.overlapping_indices) > 0:
            raise ValueError("Indices used for assignment must not overlap")

        self.build = lambda: IndexedPassThroughComp(
            expr_indices=self._tgt_indices,
            out_name=self.name,
            out_shape=self.shape,
            vals=self._tgt_vals,
        )
        self.indexed_assignment = True
        self.defined = True

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        return "Explicit Output ('" + self.name + "', " + shape_str + ")"
