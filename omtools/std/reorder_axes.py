from omtools.comps.reorder_axes_comp import ReorderAxesComp
from omtools.core.expression import Expression
from typing import List
import numpy as np
from omtools.utils.reorder_axes_utils import compute_new_axes_locations


class reorder_axes(Expression):
    def initialize(self, expr: Expression, operation: str):
        if isinstance(expr, Expression) == False:
            raise TypeError(expr, " is not an Expression object")

        self.add_predecessor_node(expr)

        # Computing out_shape
        new_axes_locations = compute_new_axes_locations(expr.shape, operation)
        self.shape = tuple(expr.shape[i] for i in new_axes_locations)

        self.build = lambda: ReorderAxesComp(
            in_name=expr.name,
            in_shape=expr.shape,
            out_name=self.name,
            out_shape=self.shape,
            operation=operation,
            new_axes_locations=new_axes_locations,
        )
