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

import builtins
from numbers import Number
from typing import List, Union

import numpy as np

from ._Matrix import Matrix
from .._types import real


def create(data: Union[np.ndarray, List[List[Number]]]) -> Matrix:
    if isinstance(data, np.ndarray):
        # TODO
        raise NotImplementedError
    else:
        return Matrix(np.array(data, real))


def eye(rows: int, columns: int = -1) -> Matrix:
    if columns == -1:
        columns = rows
    # TODO
    raise NotImplementedError


def eye_like(other: Matrix) -> Matrix:
    return eye(other.num_rows(), other.num_columns())


from_raw = create


def zeros(rows: int, columns: int) -> Matrix:
    return create([[0 for _ in builtins.range(columns)]
                   for _ in builtins.range(rows)])


def zeros_like(other: Matrix) -> Matrix:
    return zeros(other.num_rows(), other.num_columns())


def filled(rows: int, columns: int, initial_value: Number) -> Matrix:
    # TODO
    raise NotImplementedError


def filled_like(other: Matrix, initial_value: Number) -> Matrix:
    return filled(other.num_rows(), other.num_columns(), initial_value)


def from_row(vector: List[Number]) -> Matrix:
    # TODO
    raise NotImplementedError


def from_column(vector: List[Number]) -> Matrix:
    # TODO
    raise NotImplementedError


def randn(rows: int, columns: int, seed: int = 1, mean: real = real(0), std: real = real(1)) -> Matrix:
    # TODO
    raise NotImplementedError


def randn_like(other: Matrix, seed: int, mean: real = real(0), std: real = real(1)) -> Matrix:
    return randn(other.num_rows(), other.num_columns(), seed, mean, std)


def rand(rows: int, columns: int, seed: int = 1, mean: real = real(0), std: real = real(1)) -> Matrix:
    # TODO
    raise NotImplementedError


def rand_like(other: Matrix, seed: int, mean: real = real(0), std: real = real(1)) -> Matrix:
    return rand(other.num_rows(), other.num_columns(), seed, mean, std)


def diag(vector: Matrix) -> Matrix:
    n = vector.num_rows()
    result = zeros(n, n)
    for i in builtins.range(n):
        result.set(i, i, vector.get(i, 0))
    return result


def range(rows: int, columns: int, start: int) -> Matrix:
    return create([[x for x in builtins.range(s, s + columns)]
                   for s in builtins.range(start, start + rows * columns, columns)])


def create_spectrum(wave_numbers: List[Number], amplitudes: List[Number]) -> Matrix:
    if len(wave_numbers) != len(amplitudes):
        raise ValueError('Must have equal number of amplitudes and wave-numbers')

    return Matrix(np.array([[w, a] for w, a in zip(wave_numbers, amplitudes)], dtype=real))
