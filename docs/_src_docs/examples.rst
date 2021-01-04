Getting Started with Examples
=============================

The examples below will help get you started with ``omtools``.
``omtools`` provides its own ``Group`` class, which inherits from
OpenMDAO's ``Group`` class.
The ``omtools.Group`` class extends ``openmdao.Group`` so that users can
write expressions using Python syntax that ``omtools`` analyzes to
construct OpenMDAO ``Component`` objects.
The ``Component`` classes required to transform Pythonic expressions
into an OpenMDAO model with predefined derivatives are provided with
``omtools``.
The examples below include ``omtools`` expressions and n2 diagrams to
give an idea of how expressions are transformed to OpenMDAO
``Component`` objects.

.. toctree::
   :maxdepth: 2
   :titlesonly:

   examples/literals.rst
   examples/simple_explicit.rst
   examples/simple_explicit_with_subsystems.rst
   examples/ignored_outputs.rst
   examples/indep.rst
   examples/indexed_assignments_integer.rst
   examples/indexed_assignments_1d.rst
   examples/indexed_assignments_nd.rst
   examples/unary_exprs.rst
   examples/cyclic_relationships.rst
   examples/implicit_relationships.rst
   examples/implicit_with_subsystems.rst
   examples/expand.rst
