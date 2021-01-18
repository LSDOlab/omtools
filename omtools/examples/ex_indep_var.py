from omtools.api import Group
import numpy as np


class ExampleSimple(Group):
    """
    :param var: z
    """
    def setup(self):
        z = self.create_indep_var('z', val=10)

    # :param options:
    # :param bool outputs: True
    # :param bool inputs: True


# from openmdao.api import Problem

# # class is Component
# prob = Problem()
# prob.model = Group()
# prob.model.add_subsystem(<name>, ExampleSimple(<options>), promotes=['*'])
# prob.setup()
# prob.run_model()

# # class is Group
# prob = Problem()
# prob.model = ExampleSimple(<options>)
# prob.setup()
# prob.run_model()

# # print output
# print(<var>, prob['<var>'])

# # list inputs/outputs
# prob.model.list_outputs()
# prob.model.list_inputs()
