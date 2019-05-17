#  factory.py
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
Factory methods for Matrix objects.
"""

import builtins
from numbers import Number
from typing import List, Union

import numpy as np
from numpy import random

from ._Matrix import Matrix
from .._types import real


def create(data: Union[np.ndarray, List[List[Number]]]) -> Matrix:
    """
    Create a matrix from a given set of data.

    :param data:    An ndarray or a 2D list of numbers.
    :return:        A matrix.
    """
    return Matrix(np.array(data, real, copy=True))


def eye(rows: int, columns: int = -1) -> Matrix:
    """
    Creates an identity matrix (1s on the diagonal
    and 0s everywhere else.

    :param rows:        The number of rows in the matrix.
    :param columns:     The number of columns in the matrix.
                        Default is equal to rows.
    :return:            The matrix.
    """
    if columns == -1:
        columns = rows
    return Matrix(np.eye(rows, columns, dtype=real))


def eye_like(other: Matrix) -> Matrix:
    """
    Creates an identity matrix the same shape as another matrix.

    :param other:   The matrix whose dimensions to copy.
    :return:        The matrix.
    """
    return eye(other.num_rows(), other.num_columns())


from_raw = create


def zeros(rows: int, columns: int = -1) -> Matrix:
    """
    Creates a matrix filled with 0s.

    :param rows:        The number of rows the matrix should have.
    :param columns:     The number of columns the matrix should have.
                        Defaults to the number of rows.
    :return:            The matrix.
    """
    if columns == -1:
        columns = rows
    return create([[0 for _ in builtins.range(columns)]
                   for _ in builtins.range(rows)])


def zeros_like(other: Matrix) -> Matrix:
    """
    Creates a matrix filled with 0s, the same shape as the
    given matrix.

    :param other:   The matrix whose shape should be copied.
    :return:        The matrix.
    """
    return zeros(other.num_rows(), other.num_columns())


def filled(rows: int, columns: int, initial_value: Number) -> Matrix:
    """
    Creates a matrix filled with an initial value.

    :param rows:            The number of rows.
    :param columns:         The number of columns.
    :param initial_value:   The initial value to fill the matrix with.
    :return:                The matrix.
    """
    return Matrix(np.full((rows, columns), real(initial_value), real))


def filled_like(other: Matrix, initial_value: Number) -> Matrix:
    """
    Creates a matrix the same shape as another, but filled with
    an initial value.

    :param other:           The matrix whose shape should be copied.
    :param initial_value:   The value to fill the matrix with.
    :return:                The matrix.
    """
    return filled(other.num_rows(), other.num_columns(), initial_value)


def from_row(vector: List[Number]) -> Matrix:
    """
    Creates a row vector matrix from a list of values.

    :param vector:  The values to put in the matrix.
    :return:        The row vector matrix.
    """
    return create([vector])


def from_column(vector: List[Number]) -> Matrix:
    """
    Creates a column vector matrix from a list of values.

    :param vector:  The values to put in the matrix.
    :return:        The column vector matrix.
    """
    return create([[v] for v in vector])


def randn(rows: int, columns: int, seed: int = 1, mean: real = real(0), std: real = real(1)) -> Matrix:
    """
    Creates a matrix filled with random values taken from a normal distribution.

    :param rows:        The number of rows.
    :param columns:     The number of columns.
    :param seed:        The seed value to initialise the PRNG with.
    :param mean:        The mean of the normal distribution.
    :param std:         The standard deviation of the normal distribution.
    :return:            The matrix.
    """
    random.seed(seed)
    data = random.rand(rows, columns)
    np.multiply(data, std)
    np.add(data, mean)
    return Matrix(data)


def randn_like(other: Matrix, seed: int, mean: real = real(0), std: real = real(1)) -> Matrix:
    """
    Creates a matrix filled with random values taken from a normal distribution,
    in the same shape as another matrix.

    :param other:       The matrix whose shape should be copied.
    :param seed:        The seed value to initialise the PRNG with.
    :param mean:        The mean of the normal distribution.
    :param std:         The standard deviation of the normal distribution.
    :return:            The matrix.
    """
    return randn(other.num_rows(), other.num_columns(), seed, mean, std)


def rand(rows: int, columns: int, seed: int = 1) -> Matrix:
    """
    Creates a matrix filled with random values taken from a uniform distribution.

    :param rows:        The number of rows.
    :param columns:     The number of columns.
    :param seed:        The seed value to initialise the PRNG with.
    :return:            The matrix.
    """
    random.seed(seed)
    return Matrix(random.rand(rows, columns))


def rand_like(other: Matrix, seed: int) -> Matrix:
    """
    Creates a matrix filled with random values taken from a uniform distribution.

    :param other:       The matrix whose shape should be copied.
    :param seed:        The seed value to initialise the PRNG with.
    :return:            The matrix.
    """
    return rand(other.num_rows(), other.num_columns(), seed)


def diag(vector: Matrix) -> Matrix:
    """
    Creates a diagonal matrix from a column vector.

    :param vector:
    :return:
    """
    n = vector.num_rows()
    result = zeros(n, n)
    for i in builtins.range(n):
        result.set(i, i, vector.get(i, 0))
    return result


def range(rows: int, columns: int, start: int) -> Matrix:
    """
    Creates a matrix where each element is the next number in a
    sequence starting with the given value. Increments along the
    columns before wrapping to the next row.

    :param rows:        The number of rows.
    :param columns:     The number of columns.
    :param start:       The value of the first element.
    :return:            The matrix.
    """
    return create([[x for x in builtins.range(s, s + columns)]
                   for s in builtins.range(start, start + rows * columns, columns)])


def create_spectrum(wave_numbers: List[Number], amplitudes: List[Number]) -> Matrix:
    """
    Creates a spectrum matrix, which has two columns (the wave-numbers and the
    amplitudes). The provided wave-number and amplitude lists must be the same
    length.

    :param wave_numbers:    The wave-numbers of the spectrum.
    :param amplitudes:      The amplitudes of the waves.
    :return:                The spectrum matrix.
    """
    if len(wave_numbers) != len(amplitudes):
        raise ValueError('Must have equal number of amplitudes and wave-numbers')

    return Matrix(np.array([[w, a] for w, a in zip(wave_numbers, amplitudes)], dtype=real))
