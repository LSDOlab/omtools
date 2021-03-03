from typing import List, Union


def slice_to_list(
    start: Union[int, None],
    stop: Union[int, None],
    step: Union[int, None],
) -> List[int]:
    return list(
        range(
            start if start is not None else 0,
            stop if stop is not None else start + 1,
            step if step is not None else 1,
        ))
