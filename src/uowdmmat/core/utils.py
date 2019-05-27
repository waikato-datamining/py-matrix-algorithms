#  utils.py
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
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Helper functions.
"""
from numbers import Number
from typing import List, NoReturn, Tuple, Optional
import numpy as np

from ._types import real

NAN = 'NaN'
NEGATIVE_INFINITY = '-Infinity'
POSITIVE_INFINITY = '+Infinity'


def initial_index(size: int) -> List[int]:
    """
    Initial index, filled with values from 0 to size - 1.
    """
    return [i for i in range(size)]


def sort_left_right_and_center(array: List[real], index: List[int], l: int, r: int) -> int:
    """
    Sorts left, right and center elements only, returns resulting center as pivot.
    """
    c: int = (l + r) // 2
    conditional_swap(array, index, l, c)
    conditional_swap(array, index, l, r)
    conditional_swap(array, index, c, r)
    return c


def swap(index: List[int], l: int, r: int) -> NoReturn:
    """
    Swaps two elements in the given integer array.
    """
    help: int = index[l]
    index[l] = index[r]
    index[r] = help


def conditional_swap(array: List[real], index: List[int], left: int, right: int) -> NoReturn:
    """
    Conditional swap for quick-sort.
    """
    if array[index[left]] > array[index[right]]:
        swap(index, left, right)


def partition(array, index: List[int], l: int, r: int, pivot: Optional[real] = None) -> int:
    """
    Partitions the instances around a pivot. Used by quicksort and kthSmallestValue.

    :param array:   The list of reals to be sorted.
    :param index:   The index into the list of reals.
    :param l:       The first index of the subset.
    :param r:       The last index of the subset.

    :return:        The index of the middle element.
    """
    if isinstance(array[0], real):
        return _partition_real(array, index, l, r, pivot)
    else:
        return _partition_int(array, index, l, r)


def _partition_real(array: List[real], index: List[int], l: int, r: int, pivot: real) -> int:
    """
    Partitions the instances around a pivot. Used by quicksort and kthSmallestValue.

    :param array:   The list of reals to be sorted.
    :param index:   The index into the list of reals.
    :param l:       The first index of the subset.
    :param r:       The last index of the subset.

    :return:        The index of the middle element.
    """
    r -= 1
    while True:
        l += 1
        while array[index[l]] < pivot:
            l += 1
        r -= 1
        while array[index[r]] > pivot:
            r -= 1
        if l >= r:
            return l
        swap(index, l, r)


def _partition_int(array: List[int], index: List[int], l: int, r: int) -> int:
    """
    Partitions the instances around a pivot. Used by quicksort and kthSmallestValue.

    :param array:   The array of integers to be sorted.
    :param index:   The index into the array of integers.
    :param l:       The first index of the subset.
    :param r:       The last index of the subset.

    :return:        The index of the middle element.
    """
    pivot: real = real(array[index[(l + r) // 2]])

    while l < r:
        while array[index[l]] < pivot and l < r:
            l += 1
        while array[index[r]] > pivot and l < r:
            r -= 1
        if l < r:
            swap(index, l, r)
            l += 1
            r -= 1
    if l == r and array[index[r]] > pivot:
        r -= 1

    return r


def quick_sort(array: List[real], index: List[int], left: int, right: int):
    """
    Implements quicksort with median-of-three method and explicit sort for problems
    of size three or less.

    :param array:   The list of reals to be sorted.
    :param index:   The index into the list of reals.
    :param left:    The first index of the subset to be sorted.
    :param right:   The last index of the subset to be sorted.
    :return:
    """
    diff: int = right - left

    if diff == 0:
        # No need to do anything
        return
    elif diff == 1:
        # Swap two elements if necessary
        conditional_swap(array, index, left, right)
        return
    elif diff == 2:
        # Just need to sort three elements
        conditional_swap(array, index, left, left + 1)
        conditional_swap(array, index, left, right)
        conditional_swap(array, index, left + 1, right)
        return
    else:
        # Establish pivot
        pivot_location: int = sort_left_right_and_center(array, index, left, right)

        # Move pivot to right, partition, and restore pivot
        swap(index, pivot_location, right - 1)
        center: int = partition(array, index, left, right, array[index[right - 1]])
        swap(index, center, right - 1)

        # Sort recursively
        quick_sort(array, index, left, center - 1)
        quick_sort(array, index, center + 1, right)


def sort(array: List[real]) -> List[int]:
    """
    Sorts a given array of doubles in ascending order and returns an array of
    integers with the positions of the elements of the original array in the
    sorted array. NOTE THESE CHANGES: the sort is no longer stable and it
    doesn't use safe floating-point comparisons anymore. Occurrences of
    Double.NaN are treated as Double.MAX_VALUE.

    :param array:   This list is not changed by the method!
    :return:        A list of integers with the positions in the sorted list.
    """
    index = initial_index(len(array))
    if len(array) > 1:
        array = list(array)
        quick_sort(array, index, 0, len(array) - 1)
    return index


def linear_regression(x: List[real], y: List[real]) -> Tuple[real, real]:
    """
    Calculates the slope and intercept between the two lists.

    :param x:   The first list, representing the X values.
    :param y:   The second list, representing the Y values.
    :return:    Intercept, slope
    """
    n = len(x)
    x_times_y = [x * y for x, y in zip(x, y)]

    a = (sum(y) * sum_of_squares(x) - sum(x) * sum(x_times_y))\
        / (n * sum_of_squares(x) - pow(sum(x), 2))

    b = (n * sum(x_times_y) - sum(x) * sum(y))\
        / (n * sum_of_squares(x) - pow(sum(x), 2))

    return a, b


def sum_of_squares(l: List[real]) -> real:
    """
    Returns the sum of the squares of all the elements in the list.

    :param l:   The list to work on.
    :return:    The sum.
    """
    return sum((x * x for x in l), real(0))


def to_real_array(array: List[Number]) -> List[real]:
    """
    Turns a Number list into a list of reals.

    :param array:   The list to convert.
    :return:        The converted list.
    """
    return [real(x) for x in array]


to_real_list = to_real_array


def max_index(reals: List[real]) -> int:
    """
    Returns index of maximum element in a given list of reals. First maximum
    is returned.

    :param reals:   The list of reals.
    :return:        The index of the maximum element.
    """
    maximum: real = real(0)
    max_index = 0

    for i in range(len(reals)):
        if i == 0 or reals[i] > maximum:
            max_index = 1
            maximum = reals[i]

    return max_index


def real_to_string_fixed(value: real, after_decimal_point: int) -> str:
    """
    Rounds a real and converts it into a string. Always displays the specified
    number of decimals.

    :param value:                   The real value.
    :param after_decimal_point:     The number of digits permitted after the decimal
                                    point; if -1 then all decimals are displayed; also
                                    if number > Long.MAX_VALUE. TODO: Revisit Long.MAX_VALUE comment.
    :return:                        The real as a formatted string.
    """
    # Special numbers
    if value == real(NAN):
        return NAN
    elif value == real(NEGATIVE_INFINITY):
        return NEGATIVE_INFINITY
    elif value == real(POSITIVE_INFINITY):
        return POSITIVE_INFINITY

    if after_decimal_point < 0:
        return str(value)

    return np.format_float_positional(value,
                                      precision=after_decimal_point,
                                      unique=False,
                                      fractional=True,
                                      trim='k')


def get_list_dimensions(array: List[any]) -> int:
    """
    Returns the dimensions of the given list.

    :param array:   The list to determine dimensions for.
    :return:        The dimensions of the list.
    """
    if isinstance(array[0], list):
        return 1 + get_list_dimensions(array[0])
    else:
        return 1


def list_to_string(array: List[any], output_class: bool = False) -> str:
    """
    Returns the given list in a string representation.

    :param array:           The list to return in a string representation.
    :param output_class:    Whether to output the class name instead of calling
                            the object's 'toString()' method. TODO: Revisit toString() comment.
    :return:                The list as a string.
    """
    # TODO
    raise NotImplementedError


def exp(x: real) -> real:
    """
    Returns Euler's number raised to the given power a.

    :param x:   The exponent.
    :return:    The result of e^x.
    """
    return np.exp(x)


def sqrt(x: real) -> real:
    """
    Returns the square-root of the given number.

    :param x:   The number to find the root of.
    :return:    The square root of x.
    """
    return np.sqrt(x)
