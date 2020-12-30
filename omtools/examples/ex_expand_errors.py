import omtools.api as ot
from omtools.api import Group
import numpy as np


class ErrorScalarIncorrectOrder(Group):
    def setup(self):
        scalar = self.declare_input('scalar', val=1.)
        expanded_scalar = ot.expand((2, 3), scalar)
        self.register_output('expanded_scalar', expanded_scalar)


class ErrorScalarIndices(Group):
    def setup(self):
        scalar = self.declare_input('scalar', val=1.)
        expanded_scalar = ot.expand(scalar, (2, 3), [1])
        self.register_output('expanded_scalar', expanded_scalar)


class ErrorArrayNoIndices(Group):
    def setup(self):
        # Test array expansion
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_input('array', val=val)
        expanded_array = ot.expand(array, (2, 4, 3, 1))
        self.register_output('expanded_array', expanded_array)


class ErrorArrayInvalidIndices1(Group):
    def setup(self):
        # Test array expansion
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_input('array', val=val)
        expanded_array = ot.expand(array, (2, 4, 3, 1), [1])
        self.register_output('expanded_array', expanded_array)


class ErrorArrayInvalidIndices2(Group):
    def setup(self):
        # Test array expansion
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_input('array', val=val)
        expanded_array = ot.expand(array, (2, 4, 3, 1), [0, 1])
        self.register_output('expanded_array', expanded_array)