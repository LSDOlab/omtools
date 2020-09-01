from openmdao.api import ExplicitComponent, Group, Problem

from add_components_from_expressions import add_components_from_expressions
from get_variables import get_variables
from omtools.builders.cos import cos
from omtools.builders.sin import sin

class MyGroup(Group):
    def setup(self):
        # Make some variables, default shape=(1,)
        x1, x2 = get_variables(
            'x1',
            'x2',
        )

        # Store expressions
        # TODO: Literals for add, sub, mul, exprs for pow
        # y1 = 2*sin(x1**3 * x2) + x1
        # y2 = 2 * sin((x1 - 7.)**3 * x2) + x1
        # y3 = y2 + x1
        # y1 = sin(x1 + x2) + x1

        # To specify names for outputs so that they may be accessed from
        # outside the group, use the rename method
        z = cos(x1 + x2) + x1
        z.rename('z')

        add_components_from_expressions(
            self,
            x1**3 * x2,
            sin(x1**3 * x2) + x1,
            z,
        )


# a, b, c = get_variables('a', 'b', 'c')
# print((a, b, c))

# d = a + b
# print(d)

# # print(len(d.builders))
# e = d + c
# print(e)
# print(e.builders.keys())

# y = a * b
# z = d - e
# print(y)
# print(y.builders.keys())
# print(z)
# print(z.builders.keys())

# g = Group()
g = MyGroup()
# add_components_from_expressions(g, e, z, y)
prob = Problem()
prob.model = g
prob.setup()
prob.run_model()
