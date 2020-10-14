from omtools.builders.cos import cos
from omtools.builders.sin import sin
from omtools.group import Group


class MyGroup(Group):
    def setup(self):
        x1 = self.declare_input('x1')
        x2 = self.declare_input('x2')

        # Store expressions
        # y1 = 2*sin(x1**3 * x2) + x1
        # y2 = 2 * sin((x1 - 7.)**3 * x2) + x1
        # y3 = y2 + x1
        # y1 = sin(x1 + x2) + x1

        # To specify names for outputs so that they may be accessed from
        # outside the group, use the rename method
        z = cos(x1 + x2) + x1
        z.rename('z')

        self.register_output(
            'a',
            x1**3 * x2,
        )
        self.register_output(
            'b',
            sin(x1**3 * x2) + x1,
        )
        self.register_output(
            'z',
            z,
        )


if __name__ == "__main__":
    from openmdao.api import Problem

    g = MyGroup()
    prob = Problem()
    prob.model = MyGroup()
    prob.setup()
    prob.run_model()
