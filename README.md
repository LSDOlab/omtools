OM Tools
========

Introduction
------------

`omtools` (i.e. OpenMDAO Tools) provides an interface to [OpenMDAO
framework](https://openmdao.org/), making it easier for the user to
define ``Group`` subclasses by alowing the user to write expressions
without the need to define a new ``Component`` subclass or provide
analytic derivatives.
The result is that ``omtools`` ``Group`` subclasses are easier to read
and write, with assurance that analytic derivatives for their models are
correct.

Benefits
--------

- OpenMDAO ``Group`` subclass definitions are easier to read and
  write.
- Provides stock ``Component`` subclasses with pre-defined analytic
  partial derivatives.
- Prevents unnecessary feedback that can result from manually adding
  subsystems out of order.

In addition to the above benefits, ``omtools`` has a stable API, so any
improvements in efficiency for the models that ``omtools`` generates are
performance improvements for all user defined models.
This means that users can update ``omtools`` and expect performance
improvements to their models without making any changes to their models.

How it Works
------------

``omtools`` stores a graph of nodes and edges representing
expressions and their dependencies, analyzing the graph, and directing
OpenMDAO to construct corresponding ``Component`` objects and issuing
the necessary connections.
