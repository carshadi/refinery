from typing import Iterator


def range_with_end(start: int, end: int, step: int) -> Iterator[int]:
    """
    Like range, but includes the end value.

    Args:
        start: The starting value.
        end: The ending value.
        step: The step size.

    Returns:
        An iterator that yields values from start to end, inclusive.
    """
    i = start
    while i < end:
        yield i
        i += step
    yield end
