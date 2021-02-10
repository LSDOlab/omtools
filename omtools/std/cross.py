from omtools.comps.cross_product_comp import CrossProductComp
from omtools.core.expression import Expression


def cross(in1, in2, axis: int):
    if not (isinstance(in1, Expression) and isinstance(in2, Expression)):
        raise TypeError("Arguments must both be Expression objects")
    out = Expression()
    out.add_dependency_node(in1)
    out.add_dependency_node(in2)

    if in1.shape != in2.shape:
        raise Exception("The shapes of the inputs must match!")
    else:
        out.shape = in1.shape

    if in1.shape[axis] != 3:
        raise Exception(
            "The specified axis must correspond to the value of 3 in shape")

    out.build = lambda: CrossProductComp(
        shape=in1.shape,
        in1_name=in1.name,
        in2_name=in2.name,
        out_name=out.name,
        axis=axis,
    )
    return out
