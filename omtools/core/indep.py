from openmdao.api import IndepVarComp

from omtools.core.expression import Expression
from omtools.utils.get_shape_val import get_shape_val
import numpy as np


class Indep(Expression):
    """
    Container for creating a value that is constant during model
    evaluation; i.e. independent variable, or design variable
    """
    def initialize(self, name, shape=(1, ), val=1, dv=False):
        """
        Initialize independent variable

        Parameters
        ----------
        name: str
            Name of independent variable (constant during model
            evaluation)
        shape: Tuple[int]
            Shape of independent variable (constant during model
            evaluation)
        val: Number or ndarray
            Value of independent variable (constant during model
            evaluation)
        dv: bool
            Flag to set independent variable as a design variable,
            allowing the optimizer to modify the value
        """
        self.name = name
        self.shape, self.val = get_shape_val(shape, val)
        self.dv = dv

        self.build = lambda: IndepVarComp(
            name=name,
            shape=self.shape,
            val=self.val,
        )

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        return "Independent Variable ('" + self.name + "', " + shape_str + ", " + str(
            self.val) + ")"
