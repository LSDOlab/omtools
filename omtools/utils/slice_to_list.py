from typing import List, Union


def slice_to_list(
    start: Union[int, None],
    stop: Union[int, None],
    step: Union[int, None],
    size: int = None,
) -> List[int]:
    if start is None and stop is None:
        if size is None:
            raise ValueError
        else:
            stop = size
    return list(
        range(
            start if start is not None else 0,
            stop if stop is not None else start + 1,
            step if step is not None else 1,
        ))
