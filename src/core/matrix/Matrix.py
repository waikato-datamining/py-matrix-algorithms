from numbers import Number
from typing import Optional, Union, List, Tuple, Callable

import numpy as np

from core.error import MatrixAlgorithmsError
from core.matrix import real, helper


class Matrix:
    def __init__(self, data: Optional[np.ndarray] = None):
        if data is not None:
            if data.ndim > 2:
                raise ValueError('Matrix only handles 2-dimensional data!')

            while data.ndim < 2:
                data = np.array([data])

        self.data = data
        self.eigenvalue_decomposition = None
        self.singular_value_decomposition = None
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
        rows = np.array(rows, dtype=np.int)
        columns = np.array(columns, dtype=np.int)
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
        # TODO
        raise NotImplementedError

    def get_eigenvectors_sorted_ascending(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def get_eigenvectors_sorted_descending(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def get_dominant_eigenvector(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def get_eigenvalue_decomposition_V(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def get_eigenvalue_decomposition_D(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def get_eigenvalues(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def get_eigenvalues_sorted_descending(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def get_eigenvalues_sorted_ascending(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def make_singular_value_decomposition(self):
        # TODO
        raise NotImplementedError

    def make_eigenvalue_deomposition(self):
        # TODO
        raise NotImplementedError

    def make_qr_decomposition(self):
        # TODO
        raise NotImplementedError

    def svd_U(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def svd_V(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def svd_S(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def get_singular_values(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def sum(self, axis: Optional[int] = None) -> Union['Matrix', real]:
        if axis is None or axis == -1:
            return np.add.reduce(np.add.reduce(self.data))
        elif axis == 0 or axis == 1:
            result = Matrix(np.add.reduce(self.data, axis=axis))
            if axis == 1:
                result = result.transpose()
            return result
        else:
            raise ValueError('Acceptable values for axis are None, -1, 0 or 1. ' + str(axis) + 'provided.')

    def norm1(self) -> real:
        # TODO
        raise NotImplementedError

    def norm2(self) -> real:
        return np.sqrt(self.norm2_squared())

    def norm2_squared(self) -> real:
        # TODO
        raise NotImplementedError

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
        # TODO
        raise NotImplementedError

    def is_vector(self) -> bool:
        return self.data.shape[0] == 1 or self.data.shape[1] == 1

    def same_shape_as(self, other: 'Matrix') -> bool:
        return self.num_rows() == other.num_rows() and self.num_columns() == other.num_columns()

    def mul_elementwise(self, other: 'Matrix') -> 'Matrix':
        if not self.same_shape_as(other):
            helper.throw_invalid_shapes(self, other)
        return Matrix(np.multiply(self.data, other.data))

    def scale_by_row_vector(self, vector: 'Matrix') -> 'Matrix':
        helper.must_be_row_vector(vector)
        helper.dimensions_must_match(self, vector, columns_to_rows=True)
        return self.mul_by_vector(vector)

    def scale_by_column_vector(self, vector: 'Matrix') -> 'Matrix':
        helper.must_be_column_vector(vector)
        helper.dimensions_must_match(self, vector, rows_to_rows=True)
        return self.mul_by_vector(vector)

    def add_by_vector(self, vector: 'Matrix') -> 'Matrix':
        return self.vector_op(vector, np.add)

    def div_elementwise(self, other: 'Matrix') -> 'Matrix':
        if not self.same_shape_as(other):
            helper.throw_invalid_shapes(self, other)
        return Matrix(np.divide(self.data, other.data))

    def div(self, scalar: real) -> 'Matrix':
        return Matrix(np.divide(self.data, scalar))

    def sub(self, other: Union['Matrix', real]) -> 'Matrix':
        if isinstance(other, Matrix):
            helper.must_be_same_shape(self, other)
        return Matrix(np.subtract(self.data, other.data))

    def add(self, other: Union['Matrix', real]) -> 'Matrix':
        if isinstance(other, Matrix):
            helper.must_be_same_shape(self, other)
        return Matrix(np.add(self.data, other.data))

    def pow_elementwise(self, exponent: real) -> 'Matrix':
        return Matrix(np.power(self.data, exponent))

    def sqrt(self) -> 'Matrix':
        return Matrix(np.sqrt(self.data))

    def transpose(self) -> 'Matrix':
        return Matrix(self.data.transpose())

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
        helper.must_be_row_vector(row)
        helper.dimensions_must_match(self, row, columns_to_columns=True)
        self.reset_cache()
        self.data[row_index] = row.data[0]

    def set_column(self, column_index: int, column: 'Matrix'):
        helper.must_be_column_vector(column)
        helper.dimensions_must_match(self, column, rows_to_rows=True)
        self.reset_cache()
        self.data[:, column_index] = column.data[:, 0]

    def get_row(self, row_index: int) -> 'Matrix':
        return Matrix(np.array(self.data[row_index]))

    def get_column(self, column_index: int) -> 'Matrix':
        return Matrix(np.array(self.data[:, column_index])).transpose()

    def inverse(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def copy(self) -> 'Matrix':
        return Matrix(np.copy(self.data))

    def as_real(self) -> real:
        if self.num_rows() != 1 or self.num_columns() != 1:
            raise MatrixAlgorithmsError("Method Matrix#asReal is invalid " +
                                        "when number of rows != 1 or number of columns != 1.")
        return self.get(0, 0)

    def to_raw_copy_1D(self) -> List[real]:
        # TODO
        raise NotImplementedError

    def to_raw_copy_2D(self) -> List[List[real]]:
        # TODO
        raise NotImplementedError

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

    def apply_elementwise(self, body) -> 'Matrix':
        # TODO: type for body
        # TODO
        raise NotImplementedError

    def clip(self, lower_bound: real, upper_bound: real) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def clip_lower(self, lower_bound: real) -> 'Matrix':
        return self.clip(lower_bound, np.inf)

    def clip_upper(self, upper_bound: real) -> 'Matrix':
        return self.clip(-np.inf, upper_bound)

    def sign(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def abs(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def max(self) -> real:
        # TODO
        raise NotImplementedError

    def median(self) -> real:
        # TODO
        raise NotImplementedError

    def where_vector(self, condition) -> List[int]:
        # TODO: type of condition
        # TODO
        raise NotImplementedError

    def head(self, n: int = 5) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def diag(self) -> 'Matrix':
        # TODO
        raise NotImplementedError

    def mean(self, axis: int) -> real:
        # TODO
        raise NotImplementedError

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
        def element_is_NaN(el):
            return el == real('NaN')
        return self.any(element_is_NaN)

    def any(self, function: Callable[[real], bool]) -> bool:
        for row in range(self.num_rows()):
            for col in range(self.num_columns()):
                if function(self.data[row][col]):
                    return True
        return False

    def all(self, function) -> bool:
        # TODO: type for function
        # TODO
        raise NotImplementedError

    def which(self, function) -> List[Tuple[int, int]]:
        # TODO: type for function
        # TODO
        raise NotImplementedError

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

        if other is None or not isinstance(other, Matrix):
            return False

        # TODO: Complete
        raise NotImplementedError

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
        helper.must_be_vector(vector)
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

