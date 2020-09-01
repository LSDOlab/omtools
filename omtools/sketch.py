# Goal: Journal Paper
# See also rank polymorphism for LSDObox shape compatibility
# https://www.ccs.neu.edu/home/shivers/papers/rank-polymorphism.pdf
# Olin Shivers
# https://www.ccs.neu.edu/home/shivers/citations.html


def tan(x):
    return Variable(TanFunc(x))


# These classes hold expressions that can be used to create components
class TanFunc(object):
    """
    tan(arg)
    """
    def __init__(self, arg):
        self.arg = arg


class LinearFunc(object):
    """
    arg1 + coefficient * arg2
    """
    def __init__(self, arg1, arg2, coefficient):
        self.arg1 = arg1
        self.arg2 = arg2
        self.coefficient = coefficient


class PowerFunc(object):
    def __init__(self, arg, exponent):
        self.arg = arg
        self.exponent = exponent

    def __call__(self):
        return PowerComp(arg=arg)


class MyGroup(Group):
    def lsdobox_setup(self):
        # declare_variable creates a Variable object that the user can
        # access
        a = self.declare_variable('x1', shape=('num_times, num_mesh_x'))
        b = self.declare_variable('x2', shape=('num_times, num_mesh_x'))
        c = self.declare_variable('y1', shape=('num_times, num_mesh_x'))

        # The user can store expression information in a Variable object
        # Need to ensure that this overwrites the y1 variable
        # TODO: How to define implicit computations?
        # TODO: What about partial invertibility for expressions like these?
        c = lsdobox.tan(a) * b + a

        # Here we don't need to retrun instances of a Veriable because
        # we aren't going to store expression information
        self.declare_variable('a', shape=('num_times, num_mesh_x'))
        self.declare_variable('b', shape=('num_times, num_mesh_x'))

        # Create new Equation from expression; note that we are passing
        # the y1 variable that stores the expression information
        self.add_equation(Equation(c), 'name')

        # We can store as many expressions as we want in Equation
        self.add_equation(Equation(c, x, y), 'name')

        # We can also use stock Equation classes and initialize them
        # with names declared in declare_variable
        self.add_equation(
            LiftDrag(
                lift_drag='x1',
                coeff_ld='x2',
                rho='y1',
            ),
            'name',
        )

        # We can also pass other objects of Variable or
        # Variable subtype to a stock Equation
        self.add_equation(CFDUtil(cfd_solver=cfd_solver, lift='lift'))

        # Another stock Equation constructed from Variable objects
        # instead of strings
        self.add_equation(TanEquation(x1=a, x2=b))

        # Another TanEquation constructed from strings; note that a does
        # not refer to the same Variable object as 'a', and b does
        # not refer to the same Variable object as 'b'.
        self.add_equation(TanEquation(x1='a', x2='b'))

    def setup(self):
        # comp = ExecComp('y1 = 2 * tan((x1 + 1) ** 3 * x2) + x1')
        # self.add_subsystem('comp', comp, promotes=['*'])

        # Here we mimic lsdobox.Model.declare_variable; OpenMDAO doesn't
        # store/keep track of Variables in the same way as LSDObox, so
        # we must return Variable objects; we use an outsde function
        # from omtools to do this
        x1, x2, y1, y2 = omtools.get_variables('x1', 'x2', 'y1', 'y2')

        # declare the same shape for multiple Variable objects
        x1, x2, x3 = omtools.get_variables('x1', 'x2', 'x3', shape=(100))

        # default shape == (1,)
        x1, x2, x3 = omtools.get_variables('x1', 'x2', 'x3')

        # Store expressions, same as LSDObox
        y1 = sin(x1 + x2) + x1
        y2 = 2 * tan((x1 - 7.)**3 * x2) + x1

        # Create components automatically within this Group;
        # Equivalent to
        # self.add_equation(Equation(c), 'name')
        # in LSDObox API above
        self.add_auto_subsystem('group', y1, y2)

        # Now... How to deal with control flow?
        y3 = omtools.get_variable('y3', list, 10)
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            num_points_x = lifting_surface_data['num_points_x']
            num_points_z = 2 * lifting_surface_data['num_points_z_half'] - 1
            mesh_name = '{}_mesh'.format(lifting_surface_name)
            normals_name = '{}_normals'.format(lifting_surface_name)
            new = x1[2:]
            new = omtools.tan(x1)
            # FancyIndexingComp
            # LinearCombinationComp
            fr = x1[:, 0:-1, 0:-1, :]
            br = x1[:, 1:, 0:-1, :]
            fl = x1[:, 0:-1, 1:, :]
            bl = x1[:, 1:, 1:, :]
            # operations on expressions
            # only operate on certain indices
            cross = omtools.inner_product(fl - br, bl - fr, index1=2, index2=0)
            norm = compute_norm(cross)
            y3[ind] = cross / norm

        # Create components automatically within this Group as before
        om_tools.add_auto_subsystem(self, 'group', y3)
