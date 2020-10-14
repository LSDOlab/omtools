from openmdao.api import Group as OMGroup
from openmdao.api import IndepVarComp

from omtools.expression import ExplicitOutput, ImplicitOutput, Indep, Input


# https://slavik.meltser.info/convert-base-10-to-base-64-and-vise-versa-using-javascript/
def dec_to_base64(string_rep, num):
    base = 64
    char_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
    newbase = ""
    r = 0

    term = False
    while term == False:
        r = num % base
        num -= r
        num /= base
        newbase = char_set[r] + newbase
        if num > 0:
            term = True

    string_rep += newbase


class Group(OMGroup):
    def __init__(self):
        super().__init__()
        self.builders = []
        self.num_comps = 0

    def setup(self):
        """
        User defined setup
        """
        pass

    def declare_input(self, name, shape=(1, )):
        """
        Create Variable that represesnts an input to an
        ExplicitComponent
        """
        return Input(name, shape=shape)

    def create_indep_var(self, name, val=1.0, shape=(1, )):
        """
        Create Variable that represents a constant value or design
        variable; implemented using IndepVarComp
        """
        self.add_subsystem(
            name,
            IndepVarComp(name, val=val, shape=shape),
            promotes=['*'],
        )
        return Indep(name, shape=shape)

    def create_output(self, name, shape=(1, )):
        """
        Create Variable that represesnts an output of an
        ExplicitComponent; e.g. cycles, concatenation, mux/demux
        """
        return ExplicitOutput(self, name, shape=shape)

    def create_implicit_output(self, name, shape=(1, )):
        """
        Create Variable object that represesnts an output from an ImplictComponent
        """
        return ImplicitOutput(self, name, shape=shape)

    def register_output(self, name, expr):
        """
        Call builder methods to create OpenMDAO components
        """
        # The last builder object in an expression object needs to have
        # a user defined name passed to it. This name is the name of the
        # output that the builder object's build method uses to create a
        # component. For this we need to preserve ordering of builder
        # objects.
        if isinstance(expr, Input):
            raise RuntimeError("Cannot register inputs")
        if isinstance(expr, Indep):
            raise RuntimeError(
                "create_indep_var method already registered this output")
        if isinstance(expr, ExplicitOutput):
            raise RuntimeError(
                "create_output method already registered this output")
        if isinstance(expr, ImplicitOutput):
            raise RuntimeError(
                "create_implicit_output method already registered this output")

        self.builders += expr.builders
        l = len(expr.builders)
        i = 0
        # TODO: automatic name generation
        for builder in self.builders:
            if i < l - 1:
                self.add_subsystem(
                    '',
                    builder.build(),
                    promotes=['*'],
                )
            else:
                self.add_subsystem(
                    'comp_' + name,
                    builder.build(name),
                    promotes=['*'],
                )
            self.num_comps += 1
            i += 1
