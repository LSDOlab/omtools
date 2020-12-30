from omtools.core.expression import Expression
from typing import Tuple
from omtools.utils.get_shape_val import get_shape_val
import numpy as np


class Input(Expression):
    """
    Class for declaring an input variable
    """
    def initialize(
            self,
            name: str,
            shape: Tuple[int] = (1, ),
            val=1,
        # units=None,
    ):
        """
        Initialize subsystem input

        Parameters
        ----------
        name: str
            Name of variable computed by another ``System`` object
        shape: Tuple[int]
            Shape of variable computed by another ``System`` object
        val: Number or ndarray
            Default value of variable computed by another ``System``
            object, takes effect if there is no connection to another
            ``System`` object's output
        """
        self.name = name
        self.shape, self.val = get_shape_val(shape, val)
        # self.units = units

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        return "Input Variable ('" + self.name + "', " + shape_str + ", " + str(
            self.val) + ")"
