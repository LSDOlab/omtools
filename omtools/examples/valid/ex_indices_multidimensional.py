from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group


class ExampleMultidimensional(Group):
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

        # Get value from indices
        self.register_output('r', p[0, :, :])

        # Assign a vector to a slice
        vec = self.create_indep_var(
            'vec',
            shape=(1, 20),
            val=np.arange(20).reshape((1, 20)),
        )
        s = self.create_output('s', shape=(2, 20))
        s[0, :] = vec
        s[1, :] = 2 * vec


prob = Problem()
prob.model = ExampleMultidimensional()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('x', prob['x'].shape)
print(prob['x'])
print('q', prob['q'].shape)
print(prob['q'])
print('r', prob['r'].shape)
print(prob['r'])
print('s', prob['s'].shape)
print(prob['s'])
