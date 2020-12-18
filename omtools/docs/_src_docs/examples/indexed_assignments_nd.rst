Indexed Assignments (Multidimensional Indices)
==============================================

``omtools`` supports specifing ranges along multiple axes as well as
individual indices and ranges to concatenate arrays.

.. code-block:: python

  from omtools.api import Group
  from openmdao.api import Problem
  import numpy as np
  
  
  class Example(Group):
      def setup(self):
          # Works with two dimensional arrays
          z = self.declare_input('z',
                                 shape=(2, 3),
                                 val=np.arange(6).reshape((2, 3)))
          x = self.create_output('x', shape=(2, 3))
          x[0:2, 0:3] = z
  
          # Also works with higher dimensional arrays
          p = self.declare_input('p',
                                 shape=(5, 2, 3),
                                 val=np.arange(30).reshape((5, 2, 3)))
          q = self.create_output('q', shape=(5, 2, 3))
          q[0:5, 0:2, 0:3] = p
  
  
  prob = Problem()
  prob.model = Example()
  prob.setup(force_alloc_complex=True)
  prob.run_model()
  

.. embed-n2 ::
  ../examples/ex_multidimensional_index_assignment.py
