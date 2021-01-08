import numpy as np
from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver, Problem

import omtools.api as ot
# from lsdo_rotor.airfoil.quadratic_airfoil_group import QuadraticAirfoilGroup


class BEMTGroup(ot.ImplicitComponent):
    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('Cl0', default=0.)
        self.options.declare('Cl1', default=2 * np.pi)
        self.options.declare('Cd0', default=0.005)
        self.options.declare('Cd1', default=0.)
        self.options.declare('Cd2', default=0.5)
        self.options.declare('n2', default=True)

    def setup(self):
        """
        initialize will call setup and handle an exception thrown once
        finalize_options is called

        setup will be replaced with a function that
        """
        shape = self.options['shape']
        Cl0 = self.options['Cl0']
        Cl1 = self.options['Cl1']
        Cd0 = self.options['Cd0']
        Cd1 = self.options['Cd1']
        Cd2 = self.options['Cd2']

        g = self.group

        # with g.create_group('inputs_group') as group:
        group = g.create_group('inputs_group')
        group.create_indep_var('twist', val=50. * np.pi / 180.)
        group.create_indep_var('Vx', val=50)
        group.create_indep_var('Vt', val=100.)
        group.create_indep_var('sigma', val=0.15)

        phi = g.create_implicit_output('phi', shape=shape)

        Vx = g.declare_input('Vx')
        Vt = g.declare_input('Vt')
        sigma = g.declare_input('sigma')

        # group = ot.Group()
        # with g.create_group('alpha_group') as group:
        group = g.create_group('alpha_group')
        group._root.print_dag()
        twist = group.declare_input('twist')
        phi_ = group.declare_input('phi')
        alpha = twist - phi_
        group.register_output('alpha', alpha)
        group._root.print_dag()
        # g.add_subsystem('alpha_group', group)

        Cl = Cl0 + Cl1 * alpha
        Cd = Cd0 + Cd1 * alpha + Cd2 * alpha**2
        # group = QuadraticAirfoilGroup(shape=shape)
        # g.add_subsystem('airfoil_group', group, promotes=['*'])

        # Cl = g.declare_input('Cl')
        # Cd = g.declare_input('Cd')

        Cx = Cl * ot.cos(phi) - Cd * ot.sin(phi)
        Ct = Cl * ot.sin(phi) + Cd * ot.cos(phi)
        term1 = Vt * 2 * Ct * ot.sin(2 * phi) / Cx
        term2 = Vx * (2 * ot.sin(2 * phi) + Ct * sigma)
        residual = term1 - term2

        # ImplicitOutput needs knowledge of the internal Group object
        # replace leaf nodes phi with Input objects
        # register resdual as output in contained group
        # collect inputs and outputs
        phi.define_residual(residual)

        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver()
        # phi.define_residual_bracketed(
        #     residual,
        #     x1=0.,
        #     x2=np.pi / 2.,
        # )


prob = Problem()
prob.model = BEMTGroup(shape=(1, ))
prob.setup()
prob.run_model()
prob.list_outputs()
