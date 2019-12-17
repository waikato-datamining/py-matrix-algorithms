#  helper.py
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
import sys
from functools import partial
from numbers import Number
from typing import List, IO, Union

import numpy as np

from ...core.utils import real_to_string_fixed
from . import factory
from ._Matrix import Matrix
from .._types import real
from ._Axis import Axis


def delete_col(data: Matrix, col: int) -> Matrix:
    return delete_cols(data, [col])


def delete_cols(data: Matrix, cols: List[int]) -> Matrix:
    cols = [col for col in range(data.num_columns()) if col not in cols]
    return data.get_sub_matrix((0, data.num_rows()), cols)


def euclidean_distance(x: Matrix, y: Matrix, squared: bool) -> Matrix:
    xx: Matrix = row_norms(x, True)
    yy: Matrix = row_norms(y, True)

    distances: Matrix = x.matrix_multiply(y.transpose())
    distances = distances.matrix_multiply(-2)
    distances = distances.add(xx)
    distances = distances.add(yy)

    if x == y:
        fill_diagonal(distances, 0)

    if squared:
        return distances
    else:
        return distances.sqrt()


def row_norms(x: Matrix, squared: bool) -> Matrix:
    # TODO
    raise NotImplementedError


def equal(m1: Matrix, m2: Matrix, epsilon: real = real(0)) -> bool:
    if not m1.is_same_shape_as(m2):
        return False

    for row in range(m1.num_rows()):
        for col in range(m1.num_columns()):
            if np.abs(m1.get(row, col) - m2.get(row, col)) > epsilon:
                return False

    return True


def read(filename: Union[str, IO[str]], header: bool, separator: str) -> Matrix:
    """
    Reads the matrix from the given CSV file.

    :param filename:    The file to read from.
    :param header:      True if the file contains a header (gets skipped).
    :param separator:   The column separator used.
    :return:            The matrix.
    """
    if isinstance(filename, str):
        with open(filename, 'r') as file:
            lines = [line for line in file]
    else:
        lines = [line for line in filename]

    if len(lines) == 0:
        raise RuntimeError('No rows in file: ' + filename)

    if header:
        lines = lines[1:]
    if len(lines) == 0:
        raise RuntimeError('No data rows in file: ' + filename)

    sep = separator
    cells = lines[0].split(sep)
    result = factory.zeros(len(lines), len(cells))
    for i in range(len(lines)):
        cells = lines[i].split(sep)
        for j in range(min(len(cells), result.num_columns())):
            try:
                result.set(i, j, real(cells[j]))
            except Exception:
                print('Failed to parse row=' + str(i + 1 if header else i) + ' col=' + str(j) + ': ' + str(cells[j]),
                      file=sys.stderr)

    return result


def to_lines(data: Matrix, header: bool, separator: str, num_dec: int, scientific: bool = False) -> List[str]:
    result: List[str] = []

    if header:
        result.append(separator.join(['col' + str(j + 1) for j in range(data.num_columns())]))

    data_to_string(data, separator, num_dec, result, scientific)

    return result


def data_to_string(data: Matrix, separator: str, num_dec: int, result: List[str], scientific: bool):
    if scientific:
        formatter = partial(np.format_float_scientific,
                            precision=num_dec,
                            unique=True,
                            trim='0',
                            exp_digits=1)
    elif num_dec == -1:
        formatter = str
    else:
        formatter = partial(real_to_string_fixed, after_decimal_point=num_dec)

    def get_formatted(row, column):
        return formatter(data.get(row, column))

    def line_gen(row):
        return separator.join(get_formatted(row, column) for column in range(data.num_columns()))

    for row in range(data.num_rows()):
        result.append(line_gen(row))


def write(data: Matrix, filename: Union[str, IO[str]], header: bool, separator: str, num_dec: int, scientific: bool = False):
    if isinstance(filename, str):
        with open(filename, 'w') as file:
            file.writelines((line + '\n' for line in to_lines(data, header, separator, num_dec, scientific)))
    else:
        filename.writelines((line + '\n' for line in to_lines(data, header, separator, num_dec, scientific)))


def to_string(data: Matrix, header: bool = True, separator: str = '\t', num_dec: int = 6) -> str:
    return '\n'.join(to_lines(data, header, separator, num_dec))


def fill_diagonal(mat: Matrix, value: Number):
    i = 0
    while i < mat.num_rows() and i < mat.num_columns():
        mat._data[i][i] = real(value)
        i += 1


def solve(A: Matrix, b: Matrix) -> Matrix:
    return Matrix(np.linalg.solve(A._data, b._data))


def multi_concat(axis: Axis, *matrices: Matrix) -> Matrix:
    if axis is Axis.BOTH:
        raise ValueError("Must choose a specific axis for concatenation")

    return Matrix(np.concatenate([matrix._data for matrix in matrices],
                                 axis=0 if axis is Axis.ROWS else 1))
