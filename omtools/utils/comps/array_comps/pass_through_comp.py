import numpy as np
from openmdao.api import ExplicitComponent
from omtools.core.expression import Expression


class PassThroughComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('expr', types=Expression)
        self.options.declare('name', types=str)

    def setup(self):
        shape = self.options['expr'].shape
        in_name = self.options['expr'].name
        out_name = self.options['name']

        self.add_input(in_name, shape=shape)
        self.add_output(out_name, shape=shape)

        r = np.arange(np.prod(shape))
        self.declare_partials(
            out_name,
            in_name,
            val=1.,
            rows=r,
            cols=r,
        )

    def compute(self, inputs, outputs):
        in_name = self.options['expr'].name
        out_name = self.options['name']
        outputs[out_name] = inputs[in_name]


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (2, 3)

    prob = Problem()
    comp = IndepVarComp()
    comp.add_output(
        'x',
        shape=shape,
        val=np.random.rand(np.prod(shape)).reshape(shape),
    )
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = PassThroughComp(
        shape=shape,
        in_name='x',
        out_name='y',
    )
    prob.model.add_subsystem('y', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

    print(prob['x'])
    print(prob['y'])
