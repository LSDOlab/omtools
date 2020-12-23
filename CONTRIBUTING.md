# CONTRIBUTING

## Configuration Management

The `omtools` repository is hosted on GitHub and uses Git as its Source
Control Manager.

It is recommended to create a username on GitHub and
[fork](https://guides.github.com/activities/forking/) the repository.

Before committing code, be sure to write tests and make sure all tests
pass.

Please begin all commit messages with a verb in the present tense, e.g.
"update docs", not "updated docs".

After working on a feature, push your changes to your fork, and then
issue a
[pull request](https://docs.github.com/en/free-pro-team@latest/desktop/contributing-and-collaborating-using-github-desktop/creating-an-issue-or-pull-request#creating-a-pull-request)
for `omtools`.

## Contribute to Docs

`omtools` uses [Sphinx](https://www.sphinx-doc.org/en/master/) to
generate documentation automatically.
Sphinx uses `.rst` files to generate documentation.

Use `.rst` to write documentation if no example code will be included in
the page that Sphinx generates from the `.rst`.
For pages that include example code, `omtools` uses
[sphinx_auto_embed](https://github.com/hwangjt/sphinx_auto_embed) to
generate `.rst` files from `.rstx` files.
This facilitates writing documentation that includes example code and
its output.

Each time you add a feature, it is recommended to write example code
(see the `examples/` directory).
Example scripts use the `ex_` prefix by convention.
To include example code in the documentation, create a `.rstx` file
(see files in `docs/_src_docs/examples/`) describing the example code.

To generate `.rst` files from `.rstx` files, run `sphinx_auto_embed` in
the `docs/` directory.

To generate the docs, run `make html` in the `docs/` directory.

## Writing Examples

All examples are located in the `examples/` directory.
Each example script by convention uses the `ex_` prefix.

Examples should run the model when they are imported.
Do not include a `if __name__ == "__main__"` clause in your examples.

Prefer checking the partial derivatives in the tests over the examples.

## Writing Tests

`omtools` uses [pytest](https://docs.pytest.org/en/latest/) to run tests.
Tests for `Expression` subclasses are located in `tests/` and tests for
stock `Component` subclasses are located in `comps/tests/`.

All tests run an example script from `examples/`.
In order for `pytest` to collect the tests, each test suite must be
written in a file with the `test_` prefix.
Each test within a test suite is defined as a function with the `test_`
prefix as well.

> NOTE: Not all example scripts must show up in the docs, but all
> example scripts should have at least one test script.

A test suite with a single test looks as follows.

```py
from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest

def test_example_script():
    # Example script to use for test
    import omtools.examples.ex_name_of_example as example

    # Test values
    np.testing.assert_approx_equal(example.prob['var.abs.name'], desired_val)
    np.testing.assert_almost_equal(example.prob['var.abs.name'], desired_val)

    # Test partials
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)
```

> NOTE: The example script defines a `Problem`, and runs the model upon
> import. There is no `if __name__ == "__main__"` clause in the example
> script.

To test values, use
[numpy.assert_approx_equal](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_approx_equal.html)
or
[numpy.assert_almost_equal](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_almost_equal.html).

To test partial derivatives, use
`openmdao.utils.assert_utils.assert_check_partials` and
`openmdao.utils.assert_utils.assert_no_approx_partials`.

## Defining Stock Components

Stock `Component` subclasses are located in the `comps/` directory.

Stock components must provide an API that can define inputs and outputs,
including their names.

To get a feel for what's required, take a look at the `Expression`
subclasses and `Component` subclasses in `std/` and `comps/`,
respectively.

## Defining Expression Subclasses

All expressions inherit from the `Expression` class.
`Expression` subclasses are stored in the `std/` directory of `omtools`.

The `Expression` class is stored as a node on a Directed Acyclic Graph
(DAG), which `omtools` uses to determine which `Component` objects to
construct in `openmdao`, and in which order.

The two objectives when defining an `Expression` subclass are

- Establish dependence on other Expression objects
- Extract options for the corresponding stock `Component` to construct
  after ther user-defined `omtools.Group.setup` runs
- Define function that constructs

Defining an `Expression` subclass is done as folows:

```py
from omtools.core.expressin import Expression

# use snake case to make the Expression look like a function to the user
class snake_case_expression(Expression):
    def initialize(self, expr, *other_args, **kwargs):
        # First, perform error checking
        if isinstance(expr, Expression) == False
            raise TypeError(expr, " is not an Expression object")

        # Second, establish dependence of this Expression on other
        # Expression objects
        self.add_predecessor_node(expr)

        # First and second steps for variable number of Expression
        # objects
        for arg in other_args:
            if isinstance(arg, Expression) == False
                raise TypeError(arg, " is not an Expression object")
            # don't worry about an arg used multiple times
            self.add_predecessor_node(arg)

        # Third, extract options for the Component subclass that will
        # be constructed from this Expression

        # Options can come from all of the Expression objects
        # (expr, other_args) as well as named arguments (kwargs)

        # ...

        # Fourth, define the function that Group calls after user defined
        # Group.setup method is called. This function requires a name
        # argument
        self.build = lambda name : CorrespondingComponent(
            # CorrespondingComponent options defined in
            # CorrespondingComponent.initialize
        )
```

For more details, take a look at the Expression subclasses already
defined in `std/`.
