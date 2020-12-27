# OM Tools

## Introduction

`omtools` (i.e. OpenMDAO Tools) provides an interface to [OpenMDAO
framework](https://openmdao.org/), making it easier for the user to
define ``Group`` subclasses by alowing the user to write expressions
without the need to define a new ``Component`` subclass or provide
analytic derivatives.
The result is that ``omtools`` ``Group`` subclasses are easier to read
and write, with assurance that analytic derivatives for their models are
correct.

## Benefits

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

## How it Works

``omtools`` stores a graph of nodes and edges representing
expressions and their dependencies, analyzing the graph, and directing
OpenMDAO to construct corresponding ``Component`` objects and issuing
the necessary connections.

## Install

`omtools` requires Python 3.5 or later.

In Terminal, navigate to the directory where you want to keep `omtools`
files.
Then run

```sh
git clone https://github.com/lsdolab/omtools.git
cd omtools
pip install -e .
```

## Test

From the `omtools` directory, run `pytest`, and all tests will run.

```sh
cd /path/to/omtools
pytest
```

Tests are located in `tests/` and `comps/tests/`.
Running `pytest` in either of these directories will run tests for
either expressions, or stock components.

## Build Docs

To build the documentation for `omtools` locally, run

```sh
cd /path/to/omtools
cd docs/
make html
```

Then open `/path/to/omtools/docs/_build/html/index.html`.

## Developers

To contribute to `omtools`, please follow the guidelines in
[CONTRIBUTING.md](CONTRIBUTING.md).
