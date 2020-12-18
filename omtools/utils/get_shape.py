from typing import Tuple, Union
import numpy as np
from numbers import Number


def get_shape(
    shape: Tuple[int],
    val: Union[Number, np.ndarray],
) -> Tuple[int]:
    """
    Get shape from shape or value if shape is unspecified

    Parameters
    ----------
    shape: None or tuple
        Shape of value

    val: Number or ndarray
        Value

    Returns
    -------
    Tuple[int]
        Shape of value
    """
    if isinstance(val, Number):
        return shape
    if isinstance(val, np.ndarray):
        if shape == (1, ):
            return val.shape
        if val.shape != shape:
            raise ValueError("Value shape mismatch")
        return val.shape
