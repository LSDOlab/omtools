import numpy as np
from openmdao.api import ExplicitComponent

class ScalarMinComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('shape', types=tuple)
        self.options.declare('rho', types=float)

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']

        r_c = np.arange(np.prod(shape))

        self.add_input(in_name, shape=shape)

        self.add_output(out_name, shape=(1,) ) 

        self.declare_partials('*', '*', rows=1, cols= r_c)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        rho = self.options['rho']

        con_val = -inputs[in_name]
        print(con_val)

        g_max = np.min(con_val)
        print(g_max)

        g_diff = con_val - np.einsum(
            self.einsum_str,
            g_max,
            self.ones,
        )
        
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=axis)
        result = g_max + 1.0 / rho * np.log(summation)
        outputs[out_name] = result

        dsum_dg = rho * exponents
        dKS_dsum = 1.0 / (rho * np.einsum(
            self.einsum_str,
            summation,
            self.ones,
        ))
        dKS_dg = dKS_dsum * dsum_dg

        self.dKS_dg = dKS_dg

        # fmax = -inputs[in_name] - 1
        # fmax = np.maximum(fmax, -inputs[in_name])

        # arg = 0.

        # arg += np.exp(rho * (-inputs[in_name] - fmax))

        # outputs[out_name] = -(
        #     fmax + 1. / rho * np.log(arg)
        # )

    def compute_partials(self, inputs, partials):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        rho = self.options['rho']

        fmax = -inputs[in_name] - 1
        print(fmax)
        fmax = np.maximum(fmax, -inputs[in_name])

        arg = 0.
        arg += np.exp(rho * (-inputs[in_name] - fmax))


        partials[out_name, in_name] = (
                1. / arg * np.exp(rho * (-inputs[in_name] - fmax))
        ).flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, Group, IndepVarComp

    in1 = np.arange(6).reshape(2, 3)

    rho = 20.
    shape = (2, 3)

    prob = Problem()

    model = Group()

    comp = IndepVarComp()
    comp.add_output('in1', in1, shape=shape)
    model.add_subsystem('ivc', comp, promotes=['*'])

    comp = ScalarMinComp(shape=shape, in_name='in1', out_name='out', rho=rho)
    model.add_subsystem('comp', comp, promotes=['*'])

    prob.model = model
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    for var_name in ['in1', 'out']:
        print(var_name, prob[var_name])