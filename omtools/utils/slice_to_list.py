from typing import List


def slice_to_list(s: slice) -> List[int]:
    return list(range(
        s.start,
        s.stop,
        s.step if s.step is not None else 1,
    ))
