import openmdao.api as om
import omtools.api as ot
import numpy as np


class ExampleOTGroupWithinOTGroup(ot.Group):
    """
    :param var: x1
    :param var: x2
    :param var: y1

    """
    def setup(self):
        # Create independent variable
        x1 = self.create_indep_var('x1', val=40)

        # Create subsystem that depends on previously created
        # independent variable
        omtools_subgroup = ot.Group()

        # Declaring and creating variables within the omtools subgroup
        a = omtools_subgroup.declare_input('x1')
        b = omtools_subgroup.create_indep_var('x2', val=12)
        omtools_subgroup.register_output('prod', a * b)
        self.add_subsystem('omtools_subgroup',
                           omtools_subgroup,
                           promotes=['*'])

        # Declaring input that will receive its value from the omtools subgroup
        x2 = self.declare_input('x2')

        # Simple addition
        y1 = x2 + x1
        self.register_output('y1', y1)


class ExampleOTGroupWithinOMGroup(om.Group):
    """
    :param var: x1
    :param var: x2
    :param var: y1
    """
    def setup(self):
        # Create independent variable using OpenMDAO
        comp = om.IndepVarComp('x1', val=40)
        self.add_subsystem('ivc', comp, promotes=['*'])

        # Create subsystem that depends on previously created
        # independent variable
        omtools_subgroup = ot.Group()

        # Declaring and creating variables within the omtools subgroup
        a = omtools_subgroup.declare_input('x1')
        b = omtools_subgroup.create_indep_var('x2', val=12)
        omtools_subgroup.register_output('prod', a * b)
        self.add_subsystem('omtools_subgroup',
                           omtools_subgroup,
                           promotes=['*'])

        # Simple addition
        self.add_subsystem('simple_addition',
                           om.ExecComp('y1 = x2 + x1'),
                           promotes=['*'])


class ExampleOMGroupWithinOTGroup(ot.Group):
    """
    :param var: x1
    :param var: x2
    :param var: y1
    """
    def setup(self):
        # Create independent variable using Omtools
        x1 = self.create_indep_var('x1', val=40)

        # Create subsystem that depends on previously created
        # independent variable
        openmdao_subgroup = om.Group()

        # Declaring and creating variables within the omtools subgroup
        openmdao_subgroup.add_subsystem('ivc',
                                        om.IndepVarComp('x2', val=12),
                                        promotes=['*'])
        openmdao_subgroup.add_subsystem('simple_prod',
                                        om.ExecComp('prod_x1x2 = x1 * x2'),
                                        promotes=['*'])

        self.add_subsystem('openmdao_subgroup',
                           openmdao_subgroup,
                           promotes=['*'])

        # Receiving the value of x2 from the openmdao group
        x2 = self.declare_input('x2')
        # Simple addition in the Omtools group
        y1 = x2 + x1
        self.register_output('y1', y1)