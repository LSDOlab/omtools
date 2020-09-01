from omtools.expression import Variable


def get_variables(*args, shape=(1, )):
    """
    Create Variable objects from which Expressions can be built
    """
    return tuple(Variable(name, shape=shape) for name in args)
