import numpy as np
from openmdao.api import ExplicitComponent


class NormComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('in_name')
        self.options.declare('out_name')
        self.options.declare('shape')
        self.options.declare('norm_type')
        self.options.declare('axis', default=None)

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        norm_type = self.options['norm_type']
        axis = self.options['axis']

        self.rank = len(shape)

        self.add_input(
            in_name,
            shape=shape,           
        )

        if axis == None:
            self.add_output(
                out_name,
            )

        elif axis == 0:
            self.add_output(
                out_name,
                shape=(shape[1],)
            )
        
        elif axis == 1:
            self.add_output(
                out_name,
                shape=(shape[0],)
            )

        self.declare_partials(out_name, in_name)

        
    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        norm_type = self.options['norm_type']
        axis      = self.options['axis']

        if self.rank == 2:
            if axis == None:
                # Add matrix norm types below here
                if norm_type == 'fro':
                    outputs[out_name] = np.linalg.norm(inputs[in_name], 'fro')

            else:
                # Under here are axis-wise norms
                pass
                
        else:
            # Under here are all the vector norms
            if norm_type == 2:
                outputs[out_name] = np.linalg.norm(inputs[in_name], 2)      
        
        ''' Develop this later, need smoothing function to provide partial derivatives '''
        # elif norm_type == 1:
        #     outputs[out_name] = np.linalg.norm(inputs[in_name], 1)

        # elif norm_type == inf:
        #     outputs[out_name] = np.linalg.norm(inputs[in_name], np.inf)
        


    def compute_partials(self, inputs, partials):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        norm_type = self.options['norm_type']
        axis      = self.options['axis']

        if self.rank == 2:
            if axis == None:
                # Add matrix norm types below here
                if norm_type == 'fro':
                    partials[out_name, in_name] = inputs[in_name]/ np.linalg.norm(inputs[in_name], 'fro')

            else:
                # Under here are axis-wise norms
                pass
                
        else:
            # Under here are all the vector norms
            if norm_type == 2:
                partials[out_name, in_name] = inputs[in_name]/ np.linalg.norm(inputs[in_name], 2)
        

        
        # elif norm_type == 1:
        #     partials[out_name, in_name] = np.sign(inputs[in_name]) 

        # elif norm_type == inf:
        #     outputs[out_name] = np.linalg.norm(inputs[in_name], np.inf)
        


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    n = 10
    m = 20
    val = np.random.rand(n, m)
    indeps = IndepVarComp()
    indeps.add_output(
        'x',
        val=val,
        shape=(n, m),
    )
    prob = Problem()
    prob.model = Group()
    prob.model.add_subsystem(
        'indeps',
        indeps,
        promotes=['*'],
    )
    prob.model.add_subsystem(
        'fro_norm',
        MatNormComp(in_name='x', out_name='y', shape=(n, m), norm_type='fro'),
        promotes=['*'],
    )
    prob.setup()
    prob.check_partials(compact_print=True)
    prob.run_model()

 