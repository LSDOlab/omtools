import numpy as np
from openmdao.api import ExplicitComponent


class LogComp(ExplicitComponent):
    '''
    This is a component that computes the natural logarithm using numpy.log()

    Options
    -------
    in_name: str
        Name of the input 
    
    out_name: str
        Name of the output

    shape: tuple[int]
        Shape of the input and output
    '''

    def initialize(self):
        self.options.declare('in_name')
        self.options.declare('out_name')
        self.options.declare('shape')

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']

        self.add_input(
            in_name,
            shape=shape,
        )

        self.add_output(
            out_name,
            shape=shape,
        )

        r = np.arange(np.prod(shape))
        self.declare_partials(out_name, in_name, rows=r, cols=r)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        outputs[out_name] = np.log(inputs[in_name]).flatten()

    def compute_partials(self, inputs, partials):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        partials[out_name, in_name] = (1.0 / (inputs[in_name])).flatten()


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    n = 100
    shape = (n,)
    val = np.random.rand(n)
    indeps = IndepVarComp()
    indeps.add_output(
        'x',
        val=val,
        shape=shape,
    )
    prob = Problem()
    prob.model = Group()
    prob.model.add_subsystem(
        'indeps',
        indeps,
        promotes=['*'],
    )
    prob.model.add_subsystem(
        'log',
        LogComp(in_name='x', out_name='y', shape=(n, )),
        promotes=['*'],
    )
    prob.setup()
    prob.check_partials(compact_print=True)
