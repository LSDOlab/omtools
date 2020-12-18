from omtools.core.expression import Expression
from omtools.core.input import Input
from omtools.core.output import Output
from omtools.utils.replace_output_leaf_nodes import replace_output_leaf_nodes
import numpy as np
from typing import Union, Tuple, Dict, List
from omtools.utils.slice_to_list import slice_to_list
from omtools.utils.comps.array_comps.pass_through_comp import PassThroughComp
from omtools.utils.comps.array_comps.indexed_pass_through_comp import IndexedPassThroughComp
from omtools.utils.get_shape import get_shape


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
        self.shape = get_shape(shape, val)
        self.val = val
        self.defined = False
        self.indexed_assignment = False
        self.indices: Dict[Expression, List[int]] = {}
        self.checked_indices = set()
        self.overlapping_indices = set()

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

        self.add_predecessor_node(expr)
        replace_output_leaf_nodes(
            self,
            self,
            Input(self.name, shape=self.shape, val=self.val),
        )

        self.build = lambda name: PassThroughComp(
            expr=expr,
            name=name,
        )

    def __setitem__(
        self,
        key: Union[int, np.ndarray, slice, Tuple[slice]],
        expr: Expression,
    ):
        self.add_predecessor_node(expr)
        flat_dest_indices = []
        if isinstance(key, tuple):
            slices = []
            for s in key:
                slices.append(s)
            slices = [slice_to_list(s) for s in slices]
            flat_dest_indices = np.ravel_multi_index(
                tuple(np.array(np.meshgrid(*slices, indexing='ij'))),
                self.shape,
            ).flatten()
        elif isinstance(key, slice):
            flat_dest_indices = slice_to_list(key)
        elif isinstance(key, int):
            flat_dest_indices = [key]
        else:
            raise TypeError(
                "When assigning indices of an expression, key must be an int, a slice, or a tple of slices"
            )

        # Check size
        if np.amax(flat_dest_indices) >= np.prod(self.shape):
            raise ValueError("Indices given are out of range for " +
                             self.__repr__())
        self.indices[expr] = flat_dest_indices

        # Check for overlapping indices
        self.overlapping_indices = self.checked_indices.intersection(
            flat_dest_indices)
        self.checked_indices = self.checked_indices.union(flat_dest_indices)
        if len(self.overlapping_indices) > 0:
            raise ValueError("Indices used for assignment must not overlap")

        self.build = lambda name: IndexedPassThroughComp(
            expr_indices=self.indices,
            out_expr=self,
        )
        self.indexed_assignment = True
        self.defined = True

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        return "Explicit Output ('" + self.name + "', " + shape_str + ")"
