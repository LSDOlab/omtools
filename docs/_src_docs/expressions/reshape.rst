reshape
=======

This function reshapes the input to a new shape that is specified by the user. The input must be a numpy array. 

.. code-block:: python

  from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver
  from openmdao.api import Problem
  import numpy as np
  import omtools.api as ot
  from omtools.api import Group
  from omtools.core.expression import Expression
  
  
  class Example(Group):
      def setup(self):
          shape = (2, 3, 4, 5)
          size = 2 * 3 * 4 * 5
  
          # Both vector or tensors need to be numpy arrays
          tensor = np.arange(size).reshape(shape)
          vector = np.arange(size)
  
          # in1 is a tensor having shape = (2, 3, 4, 5)
          in1 = self.declare_input('in1', val=tensor)
  
          # in2 is a vector having a size of 2 * 3 * 4 * 5
          in2 = self.declare_input('in2', val=vector)
  
          # in1 is being reshaped from shape = (2, 3, 4, 5) to a vector having size = 2 * 3 * 4 * 5
          self.register_output('reshape_tensor2vector', ot.reshape(in1, new_shape= (size,) ) )
  
          # in2 is being reshaped from size =  2 * 3 * 4 * 5 to a tensor having shape = (2, 3, 4, 5)
          self.register_output('reshape_vector2tensor', ot.reshape(in2, new_shape=shape ) )
  
          
  
  prob = Problem()
  prob.model = Example()
  prob.setup(force_alloc_complex=True)
  prob.check_partials(compact_print=True)
  prob.run_model()
          
  
