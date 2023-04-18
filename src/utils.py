"""Miscellaneous utils."""


import itertools
from typing import Any, List


def first_mismatch(a: List[Any], b: List[Any], window: int = 10):
    """Returns first mismatch as well as sublists for debugging."""
    for i, (x, y) in enumerate(itertools.zip_longest(a, b)):
        if x != y:
            window_slice = slice(i - window, i + window)
            return (x, y), (a[window_slice], b[window_slice])
    return None
