import numpy as np
from openmdao.api import ExplicitComponent


class VectorizedAxisWisePnormComp(ExplicitComponent):
    """
    This is a component that computes the axis-wise p-norm of a tensor. 
    This is exclusively for p-norms that are greater than 0 and even. 

    Options
    -------
    in_name: str
        Name of the input 
    
    out_name: str
        Name of the output

    shape: tuple[int]
        Shape of the input

    pnorm_type: int
        An even integer denoting the p-norm

    axis: tuple[int]
        Represents the axis along which the p-norm is computed

    out_shape: tuple[int]
        Shape of the output after the p-norm has been taken around the axis 
    """
    def initialize(self):
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('shape', types=tuple)
        self.options.declare('pnorm_type', types=int)
        self.options.declare('axis', types=tuple)
        self.options.declare('out_shape', default= None, types=tuple)

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        pnorm_type = self.options['pnorm_type']
        axis = self.options['axis']
        out_shape = self.options['out_shape']


        self.add_input(in_name, shape=shape)


        # Computation of the einsum string that will be used in partials
        alphabet  = 'abcdefghijklmnopqrstuvwxyz'
        rank = len(shape)
        input_subscripts = alphabet[:rank]
        output_subscripts = np.delete(list(input_subscripts), axis)
        output_subscripts = ''.join(output_subscripts)

        self.operation =  '{},{}->{}'.format(
                    output_subscripts,
                    input_subscripts,
                    input_subscripts,
                )
        
        # Computation of Output shape if the shape is not provided
        if out_shape == None:
            output_shape = np.delete(shape, axis)
            self.output_shape = tuple(output_shape)
        else:
            self.output_shape = out_shape

        self.add_output(out_name, shape=self.output_shape)

        # Defining the rows and columns of the sparse partial matrix
        input_size = np.prod(shape)
        cols = np.arange(input_size)
        rows = np.unravel_index(np.arange(input_size), shape=shape)
        rows = np.delete(np.array(rows), axis, axis=0)
        rows = np.ravel_multi_index(rows, dims=self.output_shape)

        self.declare_partials(out_name, in_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        pnorm_type = self.options['pnorm_type']
        axis = self.options['axis']

        self.outputs = outputs[out_name] = np.sum(inputs[in_name] ** pnorm_type, axis=axis)**(1/pnorm_type)
    

    def compute_partials(self, inputs, partials):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        pnorm_type = self.options['pnorm_type']
        axis = self.options['axis']

        partials[out_name, in_name] = np.einsum(self.operation, self.outputs ** (1-pnorm_type), inputs[in_name] ** (pnorm_type-1)).flatten()
    


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    n = 2
    m = 3
    p = 4
    k = 5
    shape = (n,m,p,k)
    axis = (0,2)

    val = np.random.rand(n,m,p,k)
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
        'vectorized_pnorm',
        VectorizedAxisWisePnormComp(in_name='x', out_name='y',axis=axis, shape=shape, pnorm_type=8),
        promotes=['*'],
    )
    prob.setup()
    prob.check_partials(compact_print=True)
    prob.run_model()

 


# import numpy as np
# from openmdao.api import ExplicitComponent


# class VectorizedAxisWisePnormComp(ExplicitComponent):
#     def initialize(self):
#         self.options.declare('in_name')
#         self.options.declare('out_name')
#         self.options.declare('shape')
#         self.options.declare('pnorm_type')

#     def setup(self):
#         in_name = self.options['in_name']
#         out_name = self.options['out_name']
#         shape = self.options['shape']
#         pnorm_type = self.options['pnorm_type']

#         if self.rank == 2 and axis != None:
#             self.m = shape[0]       # rows
#             self.n = shape[1]       # cols


#         self.add_input(
#             in_name,
#             shape=shape,           
#         )

#         if axis == None:
#             self.add_output(
#                 out_name,
#             )

#             self.declare_partials(out_name, in_name)

#         elif axis == 0:
#             self.add_output(
#                 out_name,
#                 shape=(self.n,)
#             )

#             r = np.arange(self.n).repeat(self.m)
#             c = self.n*np.tile(np.arange(self.m),self.n) + np.arange(self.n).repeat(self.m)
#             # print=(self.n)
#             # print(self.n*np.tile(np.arange(self.m),self.n))
#             # print(c)
#             self.declare_partials(out_name, in_name, rows=r, cols=c)
        
#         elif axis == 1:
#             self.add_output(
#                 out_name,
#                 shape=(self.m,)
#             )
#             r = np.arange(self.m).repeat(self.n)
#             c = np.arange(self.m*self.n)
#             self.declare_partials(out_name, in_name, rows=r, cols=c)

        
#     def compute(self, inputs, outputs):
#         in_name = self.options['in_name']
#         out_name = self.options['out_name']
#         norm_type = self.options['norm_type']
#         axis      = self.options['axis']

#         if self.rank == 2:
#             if axis == None:
#                 # Add matrix norm types below here
#                 if norm_type == 'fro':
#                     outputs[out_name] = np.linalg.norm(inputs[in_name], 'fro')

#             else:
#                 # Under here are axis-wise norms
#                 if norm_type == 2:
#                     outputs[out_name] = np.linalg.norm(inputs[in_name], 2, axis=axis)
#                 elif norm_type == 1:
#                     outputs[out_name] = np.linalg.norm(inputs[in_name], 1, axis=axis)
#                 elif norm_type == np.inf:
#                     outputs[out_name] = np.linalg.norm(inputs[in_name], np.inf, axis=axis)

                
                
#         else:
#             # Under here are all the vector norms
#             if norm_type == 2:
#                 outputs[out_name] = np.linalg.norm(inputs[in_name], 2)      
#             elif norm_type == 1:
#                 outputs[out_name] = np.linalg.norm(inputs[in_name], 1)
#             elif norm_type == np.inf:
#                 outputs[out_name] = np.linalg.norm(inputs[in_name], np.inf)

                
        
#         ''' Develop this later, need smoothing function to provide partial derivatives '''
#         # elif norm_type == 1:
#         #     outputs[out_name] = np.linalg.norm(inputs[in_name], 1)

#         # elif norm_type == inf:
#         #     outputs[out_name] = np.linalg.norm(inputs[in_name], np.inf)
        


#     def compute_partials(self, inputs, partials):
#         in_name = self.options['in_name']
#         out_name = self.options['out_name']
#         norm_type = self.options['norm_type']
#         axis      = self.options['axis']

#         if self.rank == 2:
#             if axis == None:
#                 # Add matrix norm types below here
#                 if norm_type == 'fro':
#                     partials[out_name, in_name] = inputs[in_name]/ np.linalg.norm(inputs[in_name], 'fro')

#             elif axis == 0:
#                 # Under here are axis-wise norms
#                 if norm_type == 2:
#                     partials[out_name, in_name] = inputs[in_name].flatten('F') / np.linalg.norm(inputs[in_name], 2, axis=axis).repeat(self.m)
                
#             elif axis == 1:
#                 if norm_type == 2:
#                     partials[out_name, in_name] = inputs[in_name].flatten() / np.linalg.norm(inputs[in_name], 2, axis=axis).repeat(self.n)           
                
#         else:
#             # Under here are all the vector norms
#             if norm_type == 2:
#                 partials[out_name, in_name] = inputs[in_name]/ np.linalg.norm(inputs[in_name], 2)
        

        
#         # elif norm_type == 1:
#         #     partials[out_name, in_name] = np.sign(inputs[in_name]) 

#         # elif norm_type == inf:
#         #     outputs[out_name] = np.linalg.norm(inputs[in_name], np.inf)
        


# if __name__ == "__main__":
#     from openmdao.api import Problem, IndepVarComp, Group
#     n = 2
#     m = 3
#     val = np.random.rand(n, m)
#     indeps = IndepVarComp()
#     indeps.add_output(
#         'x',
#         val=val,
#         shape=(n, m),
#     )
#     prob = Problem()
#     prob.model = Group()
#     prob.model.add_subsystem(
#         'indeps',
#         indeps,
#         promotes=['*'],
#     )
#     prob.model.add_subsystem(
#         'fro_norm',
#         NormComp(in_name='x', out_name='y', shape=(n, m), axis=0, norm_type=2),
#         promotes=['*'],
#     )
#     prob.setup()
#     prob.check_partials(compact_print=True)
#     prob.run_model()

 