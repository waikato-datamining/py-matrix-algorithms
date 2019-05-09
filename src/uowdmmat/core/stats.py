#  stats.py
#  Copyright (C) 2019 University of Waikato, Hamilton, New Zealand
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.\

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
    return sum((x * x for x in l))
