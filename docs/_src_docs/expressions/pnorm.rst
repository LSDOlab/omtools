pnorm
=====

This is a function that computes the pnorm of a tensor. The function vectorizes the tensor input, and computes the pnorm over the 
axis/axes specified by the user. By default, if no axis/axes are specified, the pnorm of the entire tensor is computed. 

This function only supports the computation of pnorms where p is even and greater than zero. 

.. code-block:: python

  from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver
  from openmdao.api import Problem
  import numpy as np
  import omtools.api as ot
  from omtools.api import Group
  from omtools.core.expression import Expression
  
  
  class Example(Group):
      def setup(self):
          
          # Shape of the tensor
          shape = (2, 3, 4, 5)
          
          # Number of elements in the tensor
          num_of_elements = np.prod(shape)
  
          # Creating a numpy tensor with the desired shape and size
          tensor = np.arange(num_of_elements).reshape(shape)
  
          # Declaring in1 as input tensor
          in1 = self.declare_input('in1', val=tensor)
  
          # Computing the 6-norm on in1 without specifying an axis
          self.register_output('axis_free_pnorm', ot.pnorm(in1, pnorm_type=6))
  
          # Computing the 6-norm of in1 over the specified axes. 
          self.register_output('axiswise_pnorm', ot.pnorm(in1, axis=(1, 3), pnorm_type=6))
  
          
  
  prob = Problem()
  prob.model = Example()
  prob.setup(force_alloc_complex=True)
  prob.check_partials(compact_print=True)
  prob.run_model()
          
  
