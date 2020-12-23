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

        if self.rank == 2 and axis != None:
            self.m = shape[0]       # rows
            self.n = shape[1]       # cols


        self.add_input(
            in_name,
            shape=shape,           
        )

        if axis == None:
            self.add_output(
                out_name,
            )

            self.declare_partials(out_name, in_name)

        elif axis == 0:
            self.add_output(
                out_name,
                shape=(self.n,)
            )

            r = np.arange(self.n).repeat(self.m)
            c = self.n*np.tile(np.arange(self.m),self.n) + np.arange(self.n).repeat(self.m)
            # print=(self.n)
            # print(self.n*np.tile(np.arange(self.m),self.n))
            # print(c)
            self.declare_partials(out_name, in_name, rows=r, cols=c)
        
        elif axis == 1:
            self.add_output(
                out_name,
                shape=(self.m,)
            )
            r = np.arange(self.m).repeat(self.n)
            c = np.arange(self.m*self.n)
            self.declare_partials(out_name, in_name, rows=r, cols=c)

        
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
                if norm_type == 2:
                    outputs[out_name] = np.linalg.norm(inputs[in_name], 2, axis=axis)
                elif norm_type == 1:
                    outputs[out_name] = np.linalg.norm(inputs[in_name], 1, axis=axis)
                elif norm_type == np.inf:
                    outputs[out_name] = np.linalg.norm(inputs[in_name], np.inf, axis=axis)

                
                
        else:
            # Under here are all the vector norms
            if norm_type == 2:
                outputs[out_name] = np.linalg.norm(inputs[in_name], 2)      
            elif norm_type == 1:
                outputs[out_name] = np.linalg.norm(inputs[in_name], 1)
            elif norm_type == np.inf:
                outputs[out_name] = np.linalg.norm(inputs[in_name], np.inf)

                
        
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

            elif axis == 0:
                # Under here are axis-wise norms
                if norm_type == 2:
                    partials[out_name, in_name] = inputs[in_name].flatten('F') / np.linalg.norm(inputs[in_name], 2, axis=axis).repeat(self.m)
                
            elif axis == 1:
                if norm_type == 2:
                    partials[out_name, in_name] = inputs[in_name].flatten() / np.linalg.norm(inputs[in_name], 2, axis=axis).repeat(self.n)           
                
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
    n = 2
    m = 3
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
        NormComp(in_name='x', out_name='y', shape=(n, m), axis=0, norm_type=2),
        promotes=['*'],
    )
    prob.setup()
    prob.check_partials(compact_print=True)
    prob.run_model()

 