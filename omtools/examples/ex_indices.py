from omtools.api import Group
import numpy as np


class ExampleInteger(Group):
    """
    :param var: x
    """
    def setup(self):
        a = self.declare_input('a', val=0)
        b = self.declare_input('b', val=1)
        c = self.declare_input('c', val=2)
        d = self.declare_input('d', val=7.4)
        e = self.declare_input('e', val=np.pi)
        f = self.declare_input('f', val=9)
        g = e + f
        x = self.create_output('x', shape=(7, ))
        x[0] = a
        x[1] = b
        x[2] = c
        x[3] = d
        x[4] = e
        x[5] = f
        x[6] = g


class ExampleOneDimensional(Group):
    """
    :param var: x
    :param var: y
    """
    def setup(self):
        n = 20
        u = self.declare_input('u',
                               shape=(n, ),
                               val=np.arange(n).reshape((n, )))
        v = self.declare_input('v',
                               shape=(n - 4, ),
                               val=np.arange(n - 4).reshape((n - 4, )))
        w = self.declare_input('w',
                               shape=(4, ),
                               val=16 + np.arange(4).reshape((4, )))
        x = self.create_output('x', shape=(n, ))
        x[0:n] = 2 * (u + 1)
        y = self.create_output('y', shape=(n, ))
        y[0:n - 4] = 2 * (v + 1)
        y[n - 4:n] = w - 3


class ExampleMultidimensional(Group):
    """
    :param var: x
    :param var: q
    """
    def setup(self):
        # Works with two dimensional arrays
        z = self.declare_input('z',
                               shape=(2, 3),
                               val=np.arange(6).reshape((2, 3)))
        x = self.create_output('x', shape=(2, 3))
        x[0:2, 0:3] = z

        # Also works with higher dimensional arrays
        p = self.declare_input('p',
                               shape=(5, 2, 3),
                               val=np.arange(30).reshape((5, 2, 3)))
        q = self.create_output('q', shape=(5, 2, 3))
        q[0:5, 0:2, 0:3] = p


class ErrorIntegerOutOfRange(Group):
    def setup(self):
        a = self.declare_input('a', val=0)
        x = self.create_output('x', shape=(1, ))
        # This triggers an error
        x[1] = a


class ErrorIntegerOverlap(Group):
    def setup(self):
        a = self.declare_input('a', val=0)
        b = self.declare_input('b', val=1)
        x = self.create_output('x', shape=(2, ))
        x[0] = a
        # This triggers an error
        x[0] = b


class ErrorOneDimensionalOutOfRange(Group):
    def setup(self):
        n = 20
        x = self.declare_input('x',
                               shape=(n - 4, ),
                               val=np.arange(n - 4).reshape((n - 4, )))
        y = self.declare_input('y',
                               shape=(4, ),
                               val=16 + np.arange(4).reshape((4, )))
        z = self.create_output('z', shape=(n, ))
        z[0:n - 4] = 2 * (x + 1)
        # This triggers an error
        z[n - 3:n + 1] = y - 3


class ErrorOneDimensionalOverlap(Group):
    def setup(self):
        n = 20
        x = self.declare_input('x',
                               shape=(n - 4, ),
                               val=np.arange(n - 4).reshape((n - 4, )))
        y = self.declare_input('y',
                               shape=(4, ),
                               val=16 + np.arange(4).reshape((4, )))
        z = self.create_output('z', shape=(n, ))
        z[0:n - 4] = 2 * (x + 1)
        # This triggers an error
        z[n - 5:n - 1] = y - 3


class ErrorMultidimensionalOutOfRange(Group):
    def setup(self):
        z = self.declare_input('z',
                               shape=(2, 3),
                               val=np.arange(6).reshape((2, 3)))
        x = self.create_output('x', shape=(2, 3))
        # This triggers an error
        x[0:3, 0:3] = z


class ErrorMultidimensionalOverlap(Group):
    def setup(self):
        z = self.declare_input('z',
                               shape=(2, 3),
                               val=np.arange(6).reshape((2, 3)))
        x = self.create_output('x', shape=(2, 3))
        x[0:2, 0:3] = z
        # This triggers an error
        x[0:2, 0:3] = z