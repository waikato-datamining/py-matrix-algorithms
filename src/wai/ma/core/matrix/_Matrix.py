#  _Matrix.py
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

from numbers import Number
from typing import Optional, Union, List, Tuple, Callable, Set, IO

import numpy as np

from .._Serialisable import Serialisable
from ..error import MatrixAlgorithmsError, InvalidAxisError, InvalidShapeError
from .._types import real


class Matrix(Serialisable):
    def __init__(self, data: Optional[np.ndarray] = None):
        if data is not None:
            if data.ndim > 2:
                raise ValueError('Matrix only handles 2-dimensional data!')

            while data.ndim < 2:
                data = np.array([data])

        self.data = data
        self.eigenvalue_decomposition: Optional[Tuple[Matrix, Matrix]] = None
        self.singular_value_decomposition: Optional[Tuple[Matrix, Matrix, Matrix]] = None
        self.qr_decomposition = None

        # Alias
        self.t = self.transpose

    def get_sub_matrix(self,
                       rows: Union[List[int], Tuple[int, int]],
                       columns: Union[List[int], Tuple[int, int]]
                       ) -> 'Matrix':
        if isinstance(rows, tuple):
            rows = [i for i in range(*rows)]
        if isinstance(columns, tuple):
            columns = [i for i in range(*columns)]

        # Use advanced indexing
        rows = np.array(rows, dtype=int)
        columns = np.array(columns, dtype=int)
        rows = rows[:, np.newaxis]

        return Matrix(np.array(self.data[rows, columns]))

    def get_rows(self,
                 rows: Union[List[int], Tuple[int, int]]
                 ) -> 'Matrix':
        return self.get_sub_matrix(rows, (0, self.num_columns()))

    def get_columns(self,
                    columns: Union[List[int], Tuple[int, int]]
                    ) -> 'Matrix':
        return self.get_sub_matrix((0, self.num_rows()), columns)

    def get_eigenvectors(self, sort_dominance: bool = False) -> 'Matrix':
        if sort_dominance:
            return self.get_eigenvectors_sorted_descending()

        self.make_eigenvalue_decomposition()
        return self.eigenvalue_decomposition[1].copy()

    def get_eigenvectors_sorted_ascending(self) -> 'Matrix':
        return self.get_eigenvectors_sorted(True)

    def get_eigenvectors_sorted_descending(self) -> 'Matrix':
        return self.get_eigenvectors_sorted(False)

    def get_dominant_eigenvector(self) -> 'Matrix':
        return self.get_eigenvectors_sorted_descending().get_column(0)

    def get_eigenvalue_decomposition_V(self) -> 'Matrix':
        self.make_eigenvalue_decomposition()
        return self.eigenvalue_decomposition[1]

    def get_eigenvalue_decomposition_D(self) -> 'Matrix':
        self.make_eigenvalue_decomposition()
        return self.eigenvalue_decomposition[0].vector_to_diagonal()

    def get_eigenvalues(self) -> 'Matrix':
        self.make_eigenvalue_decomposition()
        return self.eigenvalue_decomposition[0].copy()

    def get_eigenvalues_sorted_descending(self) -> 'Matrix':
        return self.get_eigenvalues_sorted(False)

    def get_eigenvalues_sorted_ascending(self) -> 'Matrix':
        return self.get_eigenvalues_sorted(True)

    def make_singular_value_decomposition(self):
        u, s, vh = np.linalg.svd(self.data, full_matrices=False)
        # u and vh matrices are multiplied by -1 to match OJAlgo reference implementation
        self.singular_value_decomposition = (Matrix(u).mul(-1), Matrix(s), Matrix(vh).mul(-1))

    def make_eigenvalue_decomposition(self):
        """
        Initialises the eigenvalue decomposition.

        N.B. The numpy eigenvalue solver may produce different eigenvalues/vectors to
        the reference solver (OJAlgo in the Java matrix-algorithms repo). While these
        are still correct solutions to the decomposition, implementations that rely on
        the exact direction of the eigenvectors may be affected. Possible future solution
        is to enforce normalisation and a standard quadrant that the vectors must point into.
        """
        if self.eigenvalue_decomposition is None:
            eigenvalues, eigenvectors = np.linalg.eig(self.data)
            self.eigenvalue_decomposition = (Matrix(eigenvalues).transpose(), Matrix(eigenvectors))

    def make_qr_decomposition(self):
        # TODO
        raise NotImplementedError

    def svd_U(self) -> 'Matrix':
        self.make_singular_value_decomposition()
        return self.singular_value_decomposition[0].copy()

    def svd_V(self) -> 'Matrix':
        self.make_singular_value_decomposition()
        return self.singular_value_decomposition[2].copy()

    def svd_S(self) -> 'Matrix':
        self.make_singular_value_decomposition()
        n = self.singular_value_decomposition[1].num_elements()
        raw = [[self.singular_value_decomposition[1].get_from_vector(i) if i == j else 0
                for i in range(n)]
               for j in range(n)]
        return Matrix(np.array(raw, dtype=real))

    def get_singular_values(self) -> 'Matrix':
        return self.singular_value_decomposition[1].transpose()

    def sum(self, axis: Optional[int] = None) -> Union['Matrix', real]:
        if axis is None or axis == -1:
            result = np.add.reduce(np.add.reduce(self.data))
            if axis == -1:
                result = Matrix(result)
            return result
        elif axis == 0 or axis == 1:
            result = Matrix(np.add.reduce(self.data, axis=axis))
            if axis == 1:
                result = result.transpose()
            return result
        else:
            raise ValueError('Acceptable values for axis are None, -1, 0 or 1. ' + str(axis) + 'provided.')

    def norm1(self) -> real:
        return real(np.linalg.norm(np.linalg.norm(self.data, 1, 0), 1))

    def norm2(self) -> real:
        frobenius_norm: real = real(np.linalg.norm(self.data))
        if self.is_vector():
            return frobenius_norm
        else:
            return frobenius_norm / np.ma.sqrt(min(self.num_rows(), self.num_columns()))

    def norm2_squared(self) -> real:
        norm2 = self.norm2()
        return np.multiply(norm2, norm2)

    def mul(self, other: Union['Matrix', Number]) -> 'Matrix':
        if isinstance(other, Matrix):
            if not self.is_multiplicable_with(other):
                raise ValueError('Matrix shapes not multiplicable!')
            else:
                return Matrix(np.matmul(self.data, other.data))
        else:
            return Matrix(np.multiply(self.data, other))

    def is_multiplicable_with(self, other: 'Matrix') -> bool:
        return self.num_columns() == other.num_rows()

    def vector_dot(self, other: 'Matrix') -> real:
        if not self.is_vector() or not other.is_vector():
            raise ValueError('Both matrices must be vectors to perform vector dot operation')

        if self.data.size != other.data.size:
            raise ValueError('Both matrices must be the same length')

        if self.is_column_vector():
            a = self.data.transpose()[0]
        else:
            a = self.data[0]

        if other.is_column_vector():
            b = other.data.transpose()[0]
        else:
            b = other.data[0]

        return np.dot(a, b)

    def normalized(self, axis: int = 0) -> 'Matrix':
        norms = np.linalg.norm(self.data, 2, axis, keepdims=True)
        result = np.divide(self.data, norms)
        return Matrix(result)

    def is_vector(self) -> bool:
        return self.data.shape[0] == 1 or self.data.shape[1] == 1

    def same_shape_as(self, other: 'Matrix') -> bool:
        return self.num_rows() == other.num_rows() and self.num_columns() == other.num_columns()

    def mul_elementwise(self, other: 'Matrix') -> 'Matrix':
        if not self.same_shape_as(other):
            throw_invalid_shapes(self, other)
        return Matrix(np.multiply(self.data, other.data))

    def scale_by_row_vector(self, vector: 'Matrix') -> 'Matrix':
        must_be_column_vector(vector)
        dimensions_must_match(self, vector, columns_to_rows=True)
        return self.mul_by_vector(vector.transpose())

    def scale_by_column_vector(self, vector: 'Matrix') -> 'Matrix':
        must_be_column_vector(vector)
        dimensions_must_match(self, vector, rows_to_rows=True)
        return self.mul_by_vector(vector)

    def add_by_vector(self, vector: 'Matrix') -> 'Matrix':
        return self.vector_op(vector, np.add)

    def div_elementwise(self, other: 'Matrix') -> 'Matrix':
        if not self.same_shape_as(other):
            throw_invalid_shapes(self, other)
        return Matrix(np.divide(self.data, other.data))

    def div(self, scalar: Number) -> 'Matrix':
        return Matrix(np.divide(self.data, scalar))

    def sub(self, other: Union['Matrix', Number]) -> 'Matrix':
        if isinstance(other, Matrix):
            must_be_same_shape(self, other)
            other = other.data
        return Matrix(np.subtract(self.data, other))

    def add(self, other: Union['Matrix', Number]) -> 'Matrix':
        if isinstance(other, Matrix):
            must_be_same_shape(self, other)
            other = other.data
        return Matrix(np.add(self.data, other))

    def pow_elementwise(self, exponent: Number) -> 'Matrix':
        return Matrix(np.power(self.data, exponent))

    def sqrt(self) -> 'Matrix':
        return Matrix(np.sqrt(self.data))

    def transpose(self) -> 'Matrix':
        return Matrix(np.array(self.data.transpose(), copy=True))

    def num_columns(self) -> int:
        return self.data.shape[1]

    def num_rows(self) -> int:
        return self.data.shape[0]

    def get(self, row: int, column: int) -> real:
        return self.data[row][column]

    def set(self, row: int, column: int, value: Number):
        self.reset_cache()
        self.data[row][column] = real(value)

    def set_row(self, row_index: int, row: 'Matrix'):
        must_be_vector(row)
        if row.is_column_vector():
            row = row.transpose()
        replacement_length = min(row.num_columns(), self.num_columns())
        self.reset_cache()
        self.data[row_index, :replacement_length] = row.data[0, :replacement_length]

    def set_column(self, column_index: int, column: 'Matrix'):
        must_be_vector(column)
        if column.is_row_vector():
            column = column.transpose()
        replacement_length = min(column.num_rows(), self.num_rows())
        self.reset_cache()
        self.data[:replacement_length, column_index] = column.data[:replacement_length, 0]

    def get_row(self, row_index: int) -> 'Matrix':
        return Matrix(np.array(self.data[row_index]))

    def get_column(self, column_index: int) -> 'Matrix':
        return Matrix(np.array(self.data[:, column_index])).transpose()

    def inverse(self) -> 'Matrix':
        if self.is_square():
            return Matrix(np.linalg.inv(self.data))
        else:
            raw = [[1 if i == j else 0 for i in range(self.num_rows())] for j in range(self.num_rows())]
            rhs = np.array(raw, dtype=real)
            return Matrix(np.linalg.lstsq(self.data, rhs, rcond=None)[0])

    def copy(self) -> 'Matrix':
        return Matrix(np.copy(self.data))

    def as_real(self) -> real:
        if self.num_rows() != 1 or self.num_columns() != 1:
            raise MatrixAlgorithmsError('Method Matrix#as_real is invalid ' +
                                        'when number of rows != 1 or number of columns != 1.')
        return self.get(0, 0)

    def to_raw_copy_1D(self) -> List[real]:
        copy = []

        for col in range(self.num_columns()):
            for row in range(self.num_rows()):
                copy.append(self.get(row, col))

        return copy

    def to_raw_copy_2D(self) -> List[List[real]]:
        copy = []

        for row in range(self.num_rows()):
            r = []
            for col in range(self.num_columns()):
                r.append(self.get(row, col))
            copy.append(r)

        return copy

    def concat(self, other: 'Matrix', axis: int) -> 'Matrix':
        return Matrix(np.concatenate((self.data, other.data), axis))

    def concat_along_rows(self, other: 'Matrix') -> 'Matrix':
        return self.concat(other, 0)

    def concat_along_columns(self, other: 'Matrix') -> 'Matrix':
        return self.concat(other, 1)

    def reset_cache(self):
        self.eigenvalue_decomposition = None
        self.singular_value_decomposition = None
        self.qr_decomposition = None

    def is_row_vector(self) -> bool:
        return self.is_vector() and self.num_rows() == 1

    def is_column_vector(self) -> bool:
        return self.is_vector() and self.num_columns() == 1

    def apply_elementwise(self, body: Callable[[real], Number]) -> 'Matrix':
        result = self.copy()
        for row in range(self.num_rows()):
            for col in range(self.num_columns()):
                result.data[row][col] = real(body(result.data[row][col]))
        return result

    def clip(self, lower_bound: Number, upper_bound: Number) -> 'Matrix':
        if lower_bound > upper_bound:
            raise MatrixAlgorithmsError('Invalid clipping values. Lower ' +
                                        'bound must be below upper bound')

        lower_bound = real(lower_bound)
        upper_bound = real(upper_bound)

        def clip(el: real) -> real:
            if el < lower_bound:
                return lower_bound
            elif el > upper_bound:
                return upper_bound
            else:
                return el

        return self.apply_elementwise(clip)

    def clip_lower(self, lower_bound: Number) -> 'Matrix':
        return self.clip(lower_bound, np.inf)

    def clip_upper(self, upper_bound: Number) -> 'Matrix':
        return self.clip(-np.inf, upper_bound)

    def sign(self) -> 'Matrix':
        def sign(el: real) -> real:
            if el < 0:
                return real(-1)
            elif el > 0:
                return real(1)
            else:
                return real(0)

        return self.apply_elementwise(sign)

    def abs(self) -> 'Matrix':
        def abs(el: real) -> real:
            if el < 0:
                return -el
            else:
                return el

        return self.apply_elementwise(abs)

    def max(self) -> real:
        return np.amax(self.data)

    def median(self) -> real:
        return np.median(self.data)

    def where_vector(self, condition: Callable[[real], bool]) -> List[int]:
        must_be_vector(self)
        if self.is_row_vector():
            return [i for i in range(self.num_columns())
                    if condition(self.get(0, i))]
        else:
            return [i for i in range(self.num_rows())
                    if condition(self.get(i, 0))]

    def head(self, n: int = 5) -> 'Matrix':
        return Matrix(np.array(self.data[:n, :]))

    def diag(self) -> 'Matrix':
        raw = [[self.get(i, i)] for i in range(min(self.num_rows(), self.num_columns()))]
        return Matrix(np.array(raw))

    def mean(self, axis: Optional[int] = None) -> Union['Matrix', real]:
        if axis is None or axis == -1:
            mean = np.mean(self.data)
            if axis == -1:
                mean = Matrix(mean)
            return mean
        elif axis == 0:
            return Matrix(np.mean(self.data, axis))
        elif axis == 1:
            return Matrix(np.mean(self.data, axis)).transpose()
        else:
            raise InvalidAxisError(axis)

    def reduce_rows_L1(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def reduce_columns_L1(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def reduce_rows_L2(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def reduce_columns_L2(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def qr_Q(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def qr_R(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def contains_NaN(self) -> bool:
        NaN = real('NaN')
        def element_is_NaN(el):
            return el == NaN
        return self.any(element_is_NaN)

    def any(self, function: Callable[[real], bool]) -> bool:
        for row in range(self.num_rows()):
            for col in range(self.num_columns()):
                if function(self.data[row][col]):
                    return True
        return False

    def all(self, function: Callable[[real], bool]) -> bool:
        return not self.any(lambda el: not function(el))

    def which(self, function: Callable[[real], bool]) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        for row in range(self.num_rows()):
            for col in range(self.num_columns()):
                if function(self.data[row][col]):
                    result.append((row, col))
        return result

    def trace(self) -> real:
        return self.diag().sum()

    def __str__(self):
        string = self.row_str(0)
        for row_index in range(1, self.num_rows()):
            string += '\n' + self.row_str(row_index)
        return string

    def row_str(self, row_index: int) -> str:
        row = self.data[row_index]
        row_string = str(row[0])
        for column_index in range(1, self.num_columns()):
            row_string += ',' + str(row[column_index])
        return row_string

    def __eq__(self, other):
        if self is other:
            return True

        if other is None or self.__class__ is not other.__class__:
            return False

        diff: Matrix = self.sub(other)

        return not diff.data.any()

    def __hash__(self):
        # TODO
        raise NotImplementedError

    def shape_string(self) -> str:
        return '[' +\
               str(self.num_rows()) +\
               ' x ' +\
               str(self.num_columns()) +\
               ']'

    def sub_by_vector(self, vector: 'Matrix') -> 'Matrix':
        return self.vector_op(vector, np.subtract)

    def mul_by_vector(self, vector: 'Matrix') -> 'Matrix':
        return self.vector_op(vector, np.multiply)

    def add_by_vector_modify(self, vector: 'Matrix'):
        self.vector_op_modify(vector, np.add)

    def sub_by_vector_modify(self, vector: 'Matrix'):
        self.vector_op_modify(vector, np.subtract)

    def mul_by_vector_modify(self, vector: 'Matrix'):
        self.vector_op_modify(vector, np.multiply)

    def div_by_vector_modify(self, vector: 'Matrix'):
        self.vector_op_modify(vector, np.divide)

    def vector_op(self, vector: 'Matrix', op: np.ufunc) -> 'Matrix':
        result = self.copy()
        result.vector_op_modify(vector, op)
        return result

    def vector_op_modify(self, vector: 'Matrix', op: np.ufunc):
        must_be_vector(vector)
        self.ensure_vector_size_matches(vector)
        op.at(self.data, ..., vector.data)

    def vector_size_matches(self, vector: 'Matrix') -> bool:
        if not vector.is_vector():
            return False
        elif vector.is_row_vector():
            return self.num_columns() == vector.num_columns()
        else:
            return self.num_rows() == vector.num_rows()

    def ensure_vector_size_matches(self, vector: 'Matrix'):
        if not self.vector_size_matches(vector):
            raise ValueError('The supplied vector is not the right size')

    def num_elements(self):
        return self.num_rows() * self.num_columns()

    def get_from_vector(self, index: int):
        must_be_vector(self)
        if self.is_row_vector():
            return self.get(0, index)
        else:
            return self.get(index, 0)

    def vector_to_diagonal(self):
        must_be_vector(self)
        raw = [[self.get_from_vector(i) if i == j else real(0) for i in range(self.num_elements())]
               for j in range(self.num_elements())]
        return Matrix(np.array(raw))

    def get_eigenvectors_sorted(self, ascending: bool):
        self.make_eigenvalue_decomposition()

        # Get eigenpairs
        eigenpairs = [(self.eigenvalue_decomposition[0].get_from_vector(i),
                       self.eigenvalue_decomposition[1].get_column(i))
                      for i in range(self.num_rows())]

        eigenpairs.sort(key=lambda v: v[0], reverse=not ascending)

        first = eigenpairs[0][1]

        for i in range(1, len(eigenpairs)):
            first = first.concat_along_columns(eigenpairs[i][1])

        return first

    def get_eigenvalues_sorted(self, ascending: bool) -> 'Matrix':
        eigenvalues = self.get_eigenvalues()
        eigenvalues.data.sort(0)
        if not ascending:
            eigenvalues.data = np.flip(eigenvalues.data)
        return eigenvalues

    def exp(self):
        return Matrix(np.exp(self.data))

    def tanh(self):
        return Matrix(np.tanh(self.data))

    def is_square(self):
        return self.num_rows() == self.num_columns()

    def pseudo_inverse(self):
        return Matrix(np.linalg.pinv(self.data))

    def serialise_state(self, stream: IO[bytes]):
        # Write the number of rows and columns
        stream.write(self.serialise_to_bytes(self.num_rows()))
        stream.write(self.serialise_to_bytes(self.num_columns()))

        # Write each data element in order
        for row_index in range(self.num_rows()):
            for column_index in range(self.num_columns()):
                stream.write(self.serialise_to_bytes(self.get(row_index, column_index)))


def must_be_row_vector(vector: 'Matrix'):
    if not vector.is_row_vector():
        raise ValueError('Must be a row vector')


def must_be_column_vector(vector: 'Matrix'):
    if not vector.is_column_vector():
        raise ValueError('Must be a column vector')


def must_be_vector(vector: 'Matrix'):
    if not vector.is_vector():
        raise ValueError('Must be a vector')


def must_be_same_shape(m1: Matrix, m2: Matrix):
    dimensions_must_match(m1, m2, rows_to_rows=True, columns_to_columns=True)


def dimensions_must_match(m1: Matrix, m2: Matrix, *,
                          rows_to_rows: bool = False,
                          rows_to_columns: bool = False,
                          columns_to_rows: bool = False,
                          columns_to_columns: bool = False):
    checks: Set[bool] = {not rows_to_rows or m1.num_rows() == m2.num_rows(),
                         not rows_to_columns or m1.num_rows() == m2.num_columns(),
                         not columns_to_rows or m1.num_columns() == m2.num_rows(),
                         not columns_to_columns or m1.num_columns() == m2.num_columns()}

    if False in checks:
        raise InvalidShapeError('', m1.shape_string(), m2.shape_string())


def throw_invalid_shapes(m1: Matrix, m2: Matrix):
    raise InvalidShapeError('Invalid matrix multiplication. Shapes ' +
                            m1.shape_string() +
                            ' and ' +
                            m2.shape_string() +
                            ' do not match.')
