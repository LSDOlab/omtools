from omtools.api import Group


class ExampleUnusedInputs(Group):
    def setup(self):
        # These inputs are unused; no components will be constructed
        a = self.declare_input('a', val=10)
        b = self.declare_input('b', val=5)
        c = self.declare_input('c', val=2)


class ErrorRegisterInput(Group):
    def setup(self):
        a = self.declare_input('a', val=10)
        # This will raise a TypeError
        self.register_output('a', a)
