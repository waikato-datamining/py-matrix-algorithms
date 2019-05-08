from typing import List, Tuple

from ._types import real


def linear_regression(x: List[real], y: List[real]) -> Tuple[real, real]:
    n = len(x)
    x_times_y = [x * y for x, y in zip(x, y)]

    a = (sum(y) * sum_of_squares(x) - sum(x) * sum(x_times_y))\
        / (n * sum_of_squares(x) - pow(sum(x), 2))

    b = (n * sum(x_times_y) - sum(x) * sum(y))\
        / (n * sum_of_squares(x) - pow(sum(x), 2))

    return a, b


def sum_of_squares(l: List[real]) -> real:
    return sum([x * x for x in l])
