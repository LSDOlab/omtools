Implicit Relationships with Subsystems
======================================

Residual expressions may depend on the result of a subsystem.
For example, the solution to a quadratic equation depends on the
coefficients, but one of those coefficients may not be constant, and may
depend on a subsystem.

In this example, we solve :math:`ax^2+bx+c=0`, but :math:`a` is a fixed point of
:math:`(3 + a - 2a^2)^\frac{1}{4}`.

In order to compute :math:`a`, ``omtools`` creates a new
``openmdao.Problem`` instance within an ``ImplicitComponent``.
The ``Problem`` instance contains a model with the subsystems in each
call to ``ImplicitGroup.add_subsystem``, and computes the residual by
running ``run_model``.
**Subsystems added using ``ImplicitGroup.add_subsystem`` are part of the
residual(s), not the ``ImplicitGroup``.**

There is one restriction that ``ImplicitGroup`` places on model
definition that ``Group`` does not.
**All inputs to the ``ImplicitGroup`` must be declared using calls to
``ImplicitGroup.declare_input`` at the beginning of
``ImplicitGroup.setup``, before any calls to
``ImplicitGrou.add_subsystem``.**
In this example, the only input is `c`.
Both `a` and `b` are outputs of subsystems used to define the residual.

.. jupyter-execute::
  ../../../omtools/examples/ex_implicit_with_subsystems.py

**Note that ``ImplicitGroup`` only adds one ``ImplicitComponent`` per
call to ``ImplicitGroup.create_implicit_output``.**
Also note that calls to ``ImplicitGroup.add_subsystem`` result in
adding a subsystem to the intermal ``Problem`` instance internal to the
``ImplicitComponent`` (not shown), and not the ``ImplicitGroup`` itself.

.. embed-n2::
  ../omtools/examples/ex_implicit_with_subsystems.py

Just as with residuals that do not require subsystems to converge,
bracketing solutions is an option as well.

.. jupyter-execute::
  ../../../omtools/examples/ex_implicit_with_subsystems_bracketed_scalar.py

Brackets may also be specified for multidimensional array values.

.. jupyter-execute::
  ../../../omtools/examples/ex_implicit_with_subsystems_bracketed_array.py
