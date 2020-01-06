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
from numbers import Real
from typing import Optional, Union, List, Tuple, Callable, IO, Iterator

import numpy as np

from .._Serialisable import Serialisable
from ..errors import MatrixAlgorithmsError, InvalidShapeError
from .._types import real
from ._Axis import Axis


class Matrix(Serialisable):
    """
    Class representing a 2D matrix of real values.
    """
    def __init__(self, data: np.ndarray):
        # Make sure the data is 2-dimensional
        if data.ndim != 2:
            raise ValueError("Matrix only handles 2-dimensional data!")

        # Make sure the data is the right data type
        if data.dtype is not np.dtype(real):
            raise ValueError(f"Matrix data-type must be {real}, was {data.dtype}")

        self._data: np.ndarray = data

        # Cached decompositions of the data
        self._eigenvalue_decomposition: Optional[Tuple[Matrix, Matrix]] = None
        self._singular_value_decomposition: Optional[Tuple[Matrix, Matrix, Matrix]] = None
        self._qr_decomposition = None  # TODO: Type

    # --------- #
    # META-DATA #
    # --------- #

    def num_rows(self) -> int:
        """
        Gets the number of rows in the matrix.

        :return:    The number of rows.
        """
        return self._data.shape[0]

    def num_columns(self) -> int:
        """
        Gets the number of columns in the matrix.

        :return:    The number of columns.
        """
        return self._data.shape[1]

    def num_elements(self):
        """
        Gets the number of elements in the matrix (rows x columns).

        :return:    The number of elements.
        """
        return self._data.size

    def is_row_vector(self) -> bool:
        """
        Whether the matrix is a row-vector (has only one row).

        :return:    True if the matrix has one row.
        """
        return self._data.shape[0] == 1

    def is_column_vector(self) -> bool:
        """
        Whether the matrix is a column vector (has only one column).

        :return:    True if the matrix has one column.
        """
        return self._data.shape[1] == 1

    def is_vector(self) -> bool:
        """
        Whether the matrix is a row or column vector.

        :return:    True if the matrix is a row or column vector.
        """
        return self.is_row_vector() or self.is_column_vector()

    def is_multiplicable_with(self, other: 'Matrix') -> bool:
        """
        Whether self x other is valid for the matrices.

        :param other:   The matrix to check for compatibility.
        :return:        True if matrix multiplication is valid.
        """
        return self.num_columns() == other.num_rows()

    def is_same_shape_as(self, other: 'Matrix') -> bool:
        """
        Whether this matrix has the same number of rows and
        columns as the given matrix.

        :param other:   The matrix to compare this one with.
        :return:        True if both matrices have the same number
                        of rows and columns.
        """
        return self._data.shape == other._data.shape

    def is_square(self) -> bool:
        """
        Whether the matrix is square (same number of rows
        as columns).

        :return:    True if this matrix is square.
        """
        return self.num_rows() == self.num_columns()

    def is_scalar(self) -> bool:
        """
        Whether this matrix contains only one value.

        :return:    True if this matrix is 1x1.
        """
        return self.num_rows() == 1 and self.num_columns() == 1

    # ---------- #
    # ASSERTIONS #
    # ---------- #

    def ensure_is_vector(self):
        """
        Raises an error if this matrix is not a vector.
        """
        if not self.is_vector():
            raise ValueError("Must be a vector")

    def ensure_is_row_vector(self):
        """
        Raises an error if this matrix is not a row vector.
        """
        if not self.is_row_vector():
            raise ValueError("Must be a row vector")

    def ensure_is_column_vector(self):
        """
        Raises an error if this matrix is not a column vector.
        """
        if not self.is_column_vector():
            raise ValueError("Must be a column vector")

    def ensure_is_same_shape_as(self, other: 'Matrix'):
        """
        Raises an error if this matrix is not the same shape
        as the given matrix.

        :param other:   The matrix to compare to.
        """
        if not self.is_same_shape_as(other):
            raise InvalidShapeError("Matrices must be the same shape",
                                    self.shape_string(),
                                    other.shape_string())

    def ensure_is_multiplicable_with(self, other: 'Matrix'):
        """
        Raises an error if this matrix cannot be left-multiplied
        with the given matrix.

        :param other:    The matrix to check for compatibility.
        """
        if not self.is_multiplicable_with(other):
            raise InvalidShapeError("Matrices are not multiplicable",
                                    self.shape_string(),
                                    other.shape_string())

    # --------------- #
    # GETTERS/SETTERS #
    # --------------- #

    def get(self, row: int, column: int) -> real:
        """
        Gets the value at the given row and column.

        :param row:     The matrix row.
        :param column:  The matrix column.
        :return:        The value at the row/column.
        """
        return self._data[row][column]

    def set(self, row: int, column: int, value: Real):
        """
        Sets the value at the given row and column.

        :param row:     The matrix row.
        :param column:  The matrix column.
        :param value:   The value to set.
        """
        self._data[row][column] = real(value)
        self._reset_cache()

    def get_row(self, row_index: int) -> 'Matrix':
        """
        Gets a row from this matrix.

        :param row_index:   The row index to get.
        :return:            The row.
        """
        return Matrix(self._data[np.newaxis, row_index, :].copy())

    def get_column(self, column_index: int) -> 'Matrix':
        """
        Gets a column from this matrix.

        :param column_index:    The column index to get.
        :return:                The column.
        """
        return Matrix(self._data[:, column_index, np.newaxis].copy())

    def set_row(self, row_index: int, row: 'Matrix'):
        """
        Sets a row in this matrix to the values in the given row matrix.

        :param row_index:   The index of the row to set.
        :param row:         The values to insert into the row.
        """
        # Make sure the row is a row vector
        row.ensure_is_row_vector()

        # Make sure the row is the right length for this matrix
        row_length: int = row.num_columns()
        if row_length != self.num_columns():
            raise InvalidShapeError(f"Row doesn't have the right number of columns "
                                    f"(is {row_length}, should be {self.num_columns()})",
                                    self.shape_string(), row.shape_string())

        self._data[row_index, :] = row._data[0, :]
        self._reset_cache()

    def set_column(self, column_index: int, column: 'Matrix'):
        """
        Sets a column in this matrix to the values in the given column matrix.

        :param column_index:    The index of the column to set.
        :param column:          The values to insert into the column.
        """
        # Make sure the column is a column vector
        column.ensure_is_column_vector()

        # Make sure the column is the right length for this matrix
        column_length: int = column.num_rows()
        if column_length != self.num_rows():
            raise InvalidShapeError(f"Column doesn't have the right number of rows "
                                    f"(is {column_length}, should be {self.num_rows()})",
                                    self.shape_string(),
                                    column.shape_string())

        self._data[:, column_index] = column._data[:, 0]
        self._reset_cache()

    def get_flat(self, index: int, row_major: bool = True) -> real:
        """
        Gets a value from a matrix using a flattened index.

        :param index:       The flattened index of the element to get.
        :param row_major:   Whether to consider the index using row-major
                            ordering or column-major ordering.
        :return:            The value at the index.
        """
        # Make sure the index is in-range
        if index >= self.num_elements():
            raise IndexError(f"Flat index must be in [0, {self.num_elements()}) but was {index}")

        # Calculate the row/column index for the given ordering
        if row_major:
            row = index // self.num_columns()
            column = index % self.num_columns()
        else:
            column = index // self.num_rows()
            row = index % self.num_rows()

        return self.get(row, column)

    # ------------------ #
    # NATIVE CONVERSIONS #
    # ------------------ #

    def as_native(self) -> List[List[real]]:
        """
        Gets the data in this matrix in Python-native format.

        :return:    A list of lists of floats.
        """
        return list(self._data.tolist())

    def as_native_flattened(self) -> List[real]:
        """
        Gets the data in this matrix in flattened Python-native format.

        :return:    A list of floats.
        """
        return list(self._data.flat)

    def as_scalar(self) -> real:
        """
        Gets the sole value in this matrix if it is scalar.

        :return:    The sole matrix value.
        """
        if not self.is_scalar():
            raise InvalidShapeError("Matrix must be scalar (1x1)", self.shape_string())

        return self.get(0, 0)

    # ------------ #
    # SUB-MATRICES #
    # ------------ #

    def get_sub_matrix(self,
                       rows: Union[List[int], Tuple[int, int]],
                       columns: Union[List[int], Tuple[int, int]]) -> 'Matrix':
        """
        Gets a sub-selection of rows/columns from this matrix.

        :param rows:        The rows to get, either a list of indices or [min, max).
        :param columns:     The columns to get, either a list of indices or [min, max).
        :return:            The sub-matrix.
        """
        # Turn range-based specifications into lists
        if isinstance(rows, tuple):
            rows = [i for i in range(*rows)]
        if isinstance(columns, tuple):
            columns = [i for i in range(*columns)]

        # Use advanced indexing
        rows = np.array(rows, dtype=np.int)
        columns = np.array(columns, dtype=np.int)
        rows = rows[:, np.newaxis]

        return Matrix(np.array(self._data[rows, columns]))

    def get_rows(self, rows: Union[List[int], Tuple[int, int]]) -> 'Matrix':
        """
        Gets a sub-selection of rows from this matrix.

        :param rows:        The rows to get, either a list of indices or [min, max).
        :return:            The sub-matrix.
        """
        return self.get_sub_matrix(rows, (0, self.num_columns()))

    def get_columns(self, columns: Union[List[int], Tuple[int, int]]) -> 'Matrix':
        """
        Gets a sub-selection of columns from this matrix.

        :param columns:     The columns to get, either a list of indices or [min, max).
        :return:            The sub-matrix.
        """
        return self.get_sub_matrix((0, self.num_rows()), columns)

    def head(self, n: int = 5) -> 'Matrix':
        """
        Returns the first n rows of the matrix.

        :param n:   The number of rows to return.
        :return:    A matrix consisting of the first n rows
                    of this matrix.
        """
        return self.get_rows((0, n))

    def diag(self) -> 'Matrix':
        """
        Gets a row vector containing the diagonal elements
        of this matrix.

        :return:    The diagonal elements.
        """
        raw = [self.get(i, i) for i in range(min(self.num_rows(), self.num_columns()))]
        return Matrix(np.array([raw]))

    # ------------ #
    # AGGREGATIONS #
    # ------------ #

    def _aggregate(self, op, axis: Axis = Axis.BOTH, **kwargs) -> 'Matrix':
        """
        Base implementation of aggregation functions.

        :param op:      The aggregating function to perform.
        :param axis:    The axis to aggregate.
        :param kwargs:  Any other arguments to the aggregation op.
        :return:        The resulting matrix.
        """
        return Matrix(op(self._data,
                         axis=None if axis is Axis.BOTH else 0 if axis is Axis.COLUMNS else 1,
                         keepdims=True,
                         **kwargs))

    def maximum(self, axis: Axis = Axis.BOTH) -> 'Matrix':
        """
        Returns the maximal values in this matrix on the given axis.

        :param axis:    The axis to aggregate.
        :return:        The maxima.
        """
        return self._aggregate(np.amax, axis)

    def minimum(self, axis: Axis = Axis.BOTH) -> 'Matrix':
        """
        Returns the minimal values in this matrix along the given axis.

        :param axis:    The axis to aggregate.
        :return:        The minima.
        """
        return self._aggregate(np.amin, axis)

    def median(self, axis: Axis = Axis.BOTH) -> 'Matrix':
        """
        Returns the median values in this matrix along the given axis.

        :param axis:    The axis to aggregate.
        :return:        The medians.
        """
        return self._aggregate(np.median, axis)

    def mean(self, axis: Axis = Axis.BOTH) -> 'Matrix':
        """
        Returns the mean values in this matrix along the given axis.

        :param axis:    The axis to aggregate.
        :return:        The means.
        """
        return self._aggregate(np.mean, axis)

    def standard_deviation(self, axis: Axis = Axis.BOTH) -> 'Matrix':
        """
        Returns the standard deviations of values in this matrix along the given axis.

        :param axis:    The axis to aggregate.
        :return:        The standard deviations.
        """
        return self._aggregate(np.std, axis, ddof=1)

    def total(self, axis: Axis = Axis.BOTH) -> 'Matrix':
        """
        Returns the sums of values in this matrix along the given axis.

        :param axis:    The axis to aggregate.
        :return:        The sums.
        """
        return self._aggregate(np.sum, axis)

    def norm1(self, axis: Axis = Axis.BOTH) -> 'Matrix':
        """
        Calculates the L1 norm of values in this matrix.

        :param axis:    The axis to calculate the norm over.
        :return:        The L1 norms.
        """
        if axis is Axis.BOTH:
            norm = np.linalg.norm(np.linalg.norm(self._data, 1, 0), 1)
            return Matrix(norm.reshape((1, 1)))
        else:
            return Matrix(np.linalg.norm(self._data, 1, 0 if axis is Axis.COLUMNS else 1, keepdims=True))

    def norm2(self) -> real:
        """
        Calculates the L2 norm of this matrix.

        :return:        The L2 norm.
        """
        frobenius_norm: real = real(np.linalg.norm(self._data))
        if self.is_vector():
            return frobenius_norm
        else:
            return frobenius_norm / np.ma.sqrt(min(self.num_rows(), self.num_columns()))

    def norm2_squared(self) -> real:
        """
        Returns the square of the L2 norm of this matrix.

        :return:    The square of this matrix's L2 norm.
        """
        return np.power(self.norm2(), 2)

    def trace(self) -> real:
        """
        Gets the trace of the matrix (the sum of diagonal elements).

        :return:    The trace of the matrix.
        """
        return self.diag().total().as_scalar()

    # ------------------- #
    # EIGENVECTORS/VALUES #
    # ------------------- #

    # TODO: Comments

    def _initialise_eigenvalue_decomposition(self):
        """
        Initialises the eigenvalue decomposition.

        N.B. The numpy eigenvalue solver may produce different eigenvalues/vectors to
        the reference solver (OJAlgo in the Java matrix-algorithms repo). While these
        are still correct solutions to the decomposition, implementations that rely on
        the exact direction of the eigenvectors may be affected. Possible future solution
        is to enforce normalisation and a standard quadrant that the vectors must point into.
        """
        # Abort if already decomposed
        if self._eigenvalue_decomposition is not None:
            return

        # Perform the decomposition
        eigenvalues, eigenvectors = np.linalg.eig(self._data)

        # Convert the eigenvalues/vectors into Matrix format
        eigenvalues = eigenvalues[:, np.newaxis]
        eigenvalues = eigenvalues.astype(real)
        eigenvectors = eigenvectors.astype(real)

        # Store the decomposition
        self._eigenvalue_decomposition = (Matrix(eigenvalues), Matrix(eigenvectors))

    def get_eigenvectors(self, sort_dominance: bool = False) -> 'Matrix':
        """
        Gets the eigenvectors of this matrix.

        :param sort_dominance:  Whether to sort the vectors by
                                eigenvalue size.
        :return:                The eigenvector matrix.
        """
        # If sorted, defer
        if sort_dominance:
            return self.get_eigenvectors_sorted_descending()

        # Just return the eigenvector matrix
        return self.get_eigenvalue_decomposition_V()

    def get_eigenvectors_sorted_ascending(self) -> 'Matrix':
        """
        Gets the eigenvectors of this matrix, sorted by
        ascending eigenvalue.

        :return:    The sorted eigenvector matrix.
        """
        return self.get_eigenvectors_sorted(True)

    def get_eigenvectors_sorted_descending(self) -> 'Matrix':
        """
        Gets the eigenvectors of this matrix, sorted by
        descending eigenvalue.

        :return:    The sorted eigenvector matrix.
        """
        return self.get_eigenvectors_sorted(False)

    def get_dominant_eigenvector(self) -> 'Matrix':
        """
        Gets the eigenvector with the largest eigenvalue.

        :return:    The column eigenvector.
        """
        return self.get_eigenvectors_sorted_descending().get_column(0)

    def get_eigenvalue_decomposition_V(self) -> 'Matrix':
        """
        Gets the V-matrix of the eigen-decomposition.

        :return:    The matrix of eigenvectors.
        """
        self._initialise_eigenvalue_decomposition()
        return self._eigenvalue_decomposition[1].copy()

    def get_eigenvalue_decomposition_D(self) -> 'Matrix':
        """
        Gets the D-matrix of the eigenvalue decomposition.

        :return:    The diagonal eigenvalue matrix.
        """
        self._initialise_eigenvalue_decomposition()
        return self._eigenvalue_decomposition[0].vector_to_diagonal()

    def get_eigenvalues(self) -> 'Matrix':
        """
        Gets the eigenvalues of this matrix as a column vector.

        :return:    The eigenvalues.
        """
        self._initialise_eigenvalue_decomposition()
        return self._eigenvalue_decomposition[0].copy()

    def get_eigenvalues_sorted_descending(self) -> 'Matrix':
        """
        Gets the eigenvalues of this matrix as a column vector,
        sorted by descending magnitude.

        :return:    The eigenvalues.
        """
        return self.get_eigenvalues_sorted(False)

    def get_eigenvalues_sorted_ascending(self) -> 'Matrix':
        """
        Gets the eigenvalues of this matrix as a column vector,
        sorted by ascending magnitude.

        :return:    The eigenvalues.
        """
        return self.get_eigenvalues_sorted(True)

    def get_eigenvectors_sorted(self, ascending: bool):
        """
        Gets the eigenvectors of this matrix, sorted in either
        ascending or descending order.

        :param ascending:   The order in which to sort the eigenvectors.
        :return:            The sorted eigenvector matrix.
        """
        # Create the decomposition
        self._initialise_eigenvalue_decomposition()

        # Get eigenpairs
        eigenpairs = [(self._eigenvalue_decomposition[0].get_flat(i),
                       self._eigenvalue_decomposition[1].get_column(i))
                      for i in range(self.num_rows())]

        # Sort the eigenpairs by eigenvalue
        eigenpairs.sort(key=lambda v: v[0], reverse=not ascending)

        # Initialise the result to the first eigenvector
        result = eigenpairs[0][1]

        # Concatenate each other eigenvector in order
        for eigenvalue, eigenvector in eigenpairs[1:]:
            result = result.concatenate_along_columns(eigenvector)

        return result

    def get_eigenvalues_sorted(self, ascending: bool) -> 'Matrix':
        """
        Gets the eigenvalues of this matrix as a column vector,
        sorted by magnitude.

        :param ascending:   The order in which to sort the eigenvalues.
        :return:            The eigenvalues.
        """
        # Get the unsorted eigenvalues
        eigenvalues = self.get_eigenvalues()

        # Sort them in ascending order
        eigenvalues._data.sort(0)

        # If descending order is requested, flip the result
        if not ascending:
            eigenvalues._data = np.flip(eigenvalues._data)

        return eigenvalues

    # ---------------------------- #
    # SINGULAR VALUE DECOMPOSITION #
    # ---------------------------- #
    # TODO: Comments

    def _initialise_singular_value_decomposition(self):
        u, s, vh = np.linalg.svd(self._data, full_matrices=False)
        s.resize((1, s.size))
        # u and vh matrices are multiplied by -1 to match OJAlgo reference implementation
        self._singular_value_decomposition = (Matrix(u).multiply(-1, in_place=True),
                                              Matrix(s),
                                              Matrix(vh).multiply(-1, in_place=True))

    def svd_U(self) -> 'Matrix':
        self._initialise_singular_value_decomposition()
        return self._singular_value_decomposition[0].copy()

    def svd_V(self) -> 'Matrix':
        self._initialise_singular_value_decomposition()
        return self._singular_value_decomposition[2].copy()

    def svd_S(self) -> 'Matrix':
        self._initialise_singular_value_decomposition()
        n = self._singular_value_decomposition[1].num_elements()
        raw = [[self._singular_value_decomposition[1].get_flat(i) if i == j else 0
                for i in range(n)]
               for j in range(n)]
        return Matrix(np.array(raw, dtype=real))

    def get_singular_values(self) -> 'Matrix':
        self._initialise_singular_value_decomposition()
        return self._singular_value_decomposition[1].transpose()

    # ---------------- #
    # QR DECOMPOSITION #
    # ---------------- #
    # TODO: Comments, implementations

    def _initialise_qr_decomposition(self):
        # Abort if already decomposed
        if self._qr_decomposition is not None:
            return

        # TODO
        raise NotImplementedError

    def qr_Q(self) -> 'Matrix':
        self._initialise_qr_decomposition()

        # TODO
        raise NotImplementedError

    def qr_R(self) -> 'Matrix':
        self._initialise_qr_decomposition()

        # TODO
        raise NotImplementedError

    # ----------------- #
    # BINARY OPERATIONS #
    # ----------------- #

    def _binary_op(self, operand: Union['Matrix', Real], op: np.ufunc, in_place: bool = False) -> 'Matrix':
        """
        Performs a binary operation on this matrix's data, either
        in-place or creating a new matrix for the result.

        :param operand:     The other matrix/value to use in the operation.
        :param op:          The operation to perform.
        :param in_place:    Whether to perform this operation in place.
        :return:            The resulting matrix.
        """
        # If the operand is a matrix, make sure it is sized correctly
        if isinstance(operand, Matrix):
            if not operand.is_scalar():
                if operand.is_row_vector():
                    if self.num_columns() != operand.num_columns():
                        raise InvalidShapeError("Row vector does not match matrix shape",
                                                self.shape_string(),
                                                operand.shape_string())
                elif operand.is_column_vector():
                    if self.num_rows() != operand.num_rows():
                        raise InvalidShapeError("Column vector does not match matrix shape",
                                                self.shape_string(),
                                                operand.shape_string())
                elif not self.is_same_shape_as(operand):
                    raise InvalidShapeError("Matrix shapes don't match",
                                            self.shape_string(),
                                            operand.shape_string())

            operand = operand._data

        # Create a copy if not modifying in-place
        result = self if in_place else self.copy()

        # Perform the operation
        op.at(result._data, ..., operand)

        return result

    # TODO: Comments

    def add(self, operand: Union['Matrix', Real], in_place: bool = False) -> 'Matrix':
        """
        Adds to the values in this matrix.

        :param operand:     The values to add to this matrix. Performs differently depending on argument:
                            - scalar value or matrix: add the scalar value to all elements.
                            - row vector: add column-wise elements together.
                            - column vector: add row-wise elements together.
                            - full matrix: element-wise addition.
        :param in_place:    Whether to modify this matrix in-place.
        :return:            The result of the operation.
        """
        return self._binary_op(operand, np.add, in_place)

    def subtract(self, operand: Union['Matrix', Real], in_place: bool = False) -> 'Matrix':
        """
        Subtracts from the values in this matrix.

        :param operand:     The values to subtract from this matrix. Performs differently depending on argument:
                            - scalar value or matrix: subtract the scalar value from all elements.
                            - row vector: subtract column-wise elements from one-another.
                            - column vector: subtract row-wise elements from one-another.
                            - full matrix: element-wise subtraction.
        :param in_place:    Whether to modify this matrix in-place.
        :return:            The result of the operation.
        """
        return self._binary_op(operand, np.subtract, in_place)

    def multiply(self, operand: Union['Matrix', Real], in_place: bool = False) -> 'Matrix':
        """
        Scales the values in this matrix.

        :param operand:     The values by which to scale this matrix. Performs differently depending on argument:
                            - scalar value or matrix: scale all elements by the value.
                            - row vector: multiply column-wise elements with one-another.
                            - column vector: multiply row-wise elements with one-another.
                            - full matrix: element-wise multiplication.
        :param in_place:    Whether to modify this matrix in-place.
        :return:            The result of the operation.
        """
        return self._binary_op(operand, np.multiply, in_place)

    def divide(self, operand: Union['Matrix', Real], in_place: bool = False) -> 'Matrix':
        """
        Divides the values in this matrix.

        :param operand:     The values by which to divide this matrix. Performs differently depending on argument:
                            - scalar value or matrix: divide all elements by the value.
                            - row vector: divide column-wise elements by one-another.
                            - column vector: divide row-wise elements by one-another.
                            - full matrix: element-wise division.
        :param in_place:    Whether to modify this matrix in-place.
        :return:            The result of the operation.
        """
        return self._binary_op(operand, np.divide, in_place)

    def pow(self, operand: Union['Matrix', Real], in_place: bool = False) -> 'Matrix':
        """
        Raises the values in this matrix to the given exponents.

        :param operand:     The values to exponentiate this matrix. Performs differently depending on argument:
                            - scalar value or matrix: exponentiate all elements by this value.
                            - row vector: exponentiate column-wise elements.
                            - column vector: exponentiate row-wise elements.
                            - full matrix: element-wise exponentiation.
        :param in_place:    Whether to modify this matrix in-place.
        :return:            The result of the operation.
        """
        return self._binary_op(operand, np.power, in_place)

    def vector_dot(self, other: 'Matrix') -> real:
        """
        Calculates the vector dot-product of two vector matrices.

        :param other:   The other vector matrix to combine with this one.
        :return:        The vector dot-product of the two matrices.
        """
        # Both matrices must be vectors
        if not self.is_vector() or not other.is_vector():
            raise ValueError('Both matrices must be vectors to perform vector dot operation')

        # Both vectors must be the same length
        if self._data.size != other._data.size:
            raise ValueError('Both matrices must be the same length')

        # Extract the vector data for Numpy
        a: np.ndarray = (self._data.transpose() if self.is_column_vector() else self._data)[0]
        b: np.ndarray = (other._data.transpose() if other.is_column_vector() else other._data)[0]

        return real(np.dot(a, b))

    def matrix_multiply(self, other: 'Matrix', in_place: bool = False) -> 'Matrix':
        """
        Performs matrix multiplication between this and another matrix.

        :param other:       The matrix to multiply with this one.
        :param in_place:    Whether the result should be stored in this matrix,
                            or a new matrix should be created.
        :return:            The result of the multiplication.
        """
        # Make sure the multiplication is valid
        self.ensure_is_multiplicable_with(other)

        # Perform the multiplication
        if in_place:
            self._data = np.matmul(self._data, other._data)
            return self
        else:
            return Matrix(np.matmul(self._data, other._data))

    def clip(self,
             lower_bound: Optional[Real] = None,
             upper_bound: Optional[Real] = None,
             in_place: bool = False) -> 'Matrix':
        """
        Returns a matrix with all values clipped to the provided bounds.

        :param lower_bound:     The minimum value that should be in the matrix.
        :param upper_bound:     The maximum value that should be in the matrix.
        :param in_place:        Whether the operation should be done on this matrix,
                                or a new matrix should be returned.
        :return:                The clipped matrix.
        """
        # If both bounds are none, no clipping is performed
        if lower_bound is None and upper_bound is None:
            return self if in_place else self.copy()

        # Make sure the bounds are properly ordered
        if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
            raise MatrixAlgorithmsError(f"Lower bound {lower_bound} must be below upper bound {upper_bound}")

        # Determine the resulting matrix
        if in_place:
            np.clip(self._data, lower_bound, upper_bound, self._data)
            return self
        else:
            return Matrix(np.clip(self._data, lower_bound, upper_bound))

    # ---------------- #
    # UNARY OPERATIONS #
    # ---------------- #
    # TODO: Comments

    def _unary_op(self, op: np.ufunc, in_place: bool = False) -> 'Matrix':
        result = self if in_place else self.copy()

        op.at(result._data, ...)

        return result

    def sign(self, in_place: bool = False) -> 'Matrix':
        return self._unary_op(np.sign, in_place)

    def abs(self, in_place: bool = False) -> 'Matrix':
        return self._unary_op(np.abs, in_place)

    def log(self, in_place: bool = False) -> 'Matrix':
        return self._unary_op(np.log, in_place)

    def exp(self, in_place: bool = False) -> 'Matrix':
        return self._unary_op(np.exp, in_place)

    def tanh(self, in_place: bool = False) -> 'Matrix':
        return self._unary_op(np.tanh, in_place)

    def sqrt(self, in_place: bool = False) -> 'Matrix':
        return self._unary_op(np.sqrt, in_place)

    def normalized(self, axis: Axis = Axis.BOTH) -> 'Matrix':
        """
        TODO
        :param axis:
        :return:
        """
        norms = np.linalg.norm(self._data, 2,
                               None if axis is Axis.BOTH else 0 if axis is Axis.COLUMNS else 1,
                               keepdims=True)
        result = np.divide(self._data, norms)
        return Matrix(result)

    def apply_elementwise(self, function: Callable[[real], Real], in_place: bool = False) -> 'Matrix':
        """
        Applies a function to each element of a matrix.

        :param function:    The function to apply to each element.
        :param in_place:    Whether this matrix should be modified in-place.
        :return:            The resulting matrix.
        """
        # Determine the target
        result = self if in_place else self.copy()

        # Apply the function to each element
        for row, column, value in self.evaluate(self.row_major_iterator()):
            result._data[row][column] = real(function(value))

        # Reset the cache
        result._reset_cache()

        return result

    # ------- #
    # INVERSE #
    # ------- #
    # TODO: Comments

    def inverse(self) -> 'Matrix':
        if self.is_square():
            return Matrix(np.linalg.inv(self._data))
        else:
            raw = [[1 if i == j else 0 for i in range(self.num_rows())] for j in range(self.num_rows())]
            rhs = np.array(raw, dtype=real)
            return Matrix(np.linalg.lstsq(self._data, rhs, rcond=None)[0])

    def pseudo_inverse(self):
        return Matrix(np.linalg.pinv(self._data))

    # ------------- #
    # CONCATENATION #
    # ------------- #

    def concatenate(self, other: 'Matrix', axis: Axis) -> 'Matrix':
        """
        Concatenates this matrix with the other along the
        given axis.

        :param other:   The matrix to concatenate with.
        :param axis:    The axis along which to concatenate.
        :return:        The resulting concatenated matrix.
        """
        # Axis must be specific
        if axis is Axis.BOTH:
            raise ValueError("Can't concatenate along both axes")

        return Matrix(np.concatenate((self._data, other._data),
                                     axis=0 if axis is Axis.ROWS else 1))

    def concatenate_along_rows(self, other: 'Matrix') -> 'Matrix':
        """
        Concatenates this matrix with the other along their rows.

        :param other:   The matrix to concatenate with.
        :return:        The resulting concatenated matrix.
        """
        return self.concatenate(other, Axis.ROWS)

    def concatenate_along_columns(self, other: 'Matrix') -> 'Matrix':
        """
        Concatenates this matrix with the other along their columns.

        :param other:   The matrix to concatenate with.
        :return:        The resulting concatenated matrix.
        """
        return self.concatenate(other, Axis.COLUMNS)

    # ---------- #
    # PREDICATES #
    # ---------- #

    def any(self, predicate: Callable[[real], bool]) -> bool:
        """
        Whether any element in this matrix matches the given
        predicate.

        :param predicate:   A functional predicate to test.
        :return:            True if any value matches.
        """
        return any(map(predicate, self.evaluate_only(self.row_major_iterator())))

    def all(self, predicate: Callable[[real], bool]) -> bool:
        """
        Whether all elements in this matrix match the given
        predicate.

        :param predicate:   A functional predicate to test.
        :return:            True if all values match.
        """
        return all(map(predicate, self.evaluate_only(self.row_major_iterator())))

    def where(self, predicate: Callable[[real], bool]) -> List[Tuple[int, int]]:
        """
        Returns the (row, column) indices of all elements
        that match the given predicate.

        :param predicate:   The functional predicate to match.
        :return:            A list of (row, column) pairs of matching elements.
        """
        return [(row, column)
                for row, column, value in self.evaluate(self.row_major_iterator())
                if predicate(value)]

    def where_vector(self, predicate: Callable[[real], bool]) -> List[int]:
        """
        Returns single-valued indices for elements in a vector which
        match the given predicate (row-index for column vectors and
        column-index for row vectors).

        :param predicate:   The functional predicate to match.
        :return:            A list of indices of matching elements.
        """
        # Can only be called for vectors
        self.ensure_is_vector()

        if self.is_row_vector():
            return [column for row, column in self.where(predicate)]
        else:
            return [row for row, column in self.where(predicate)]

    def contains_NaN(self) -> bool:
        """
        Whether any of the values in the matrix are not-a-number.

        :return:    True if any element is NaN.
        """
        return self.any(lambda value: value == np.NaN)

    # --------- #
    # ITERATION #
    # --------- #

    def row_major_iterator(self) -> Iterator[Tuple[int, int]]:
        """
        Returns an iterator over the indices of this matrix in
        row-major order.

        :return:    The iterator of (row, column) pairs.
        """
        return ((row, column)
                for column in range(self.num_columns())
                for row in range(self.num_rows()))

    def column_major_iterator(self) -> Iterator[Tuple[int, int]]:
        """
        Returns an iterator over the indices of this matrix in
        column-major order.

        :return:    The iterator of (row, column) pairs.
        """
        return ((row, column)
                for row in range(self.num_rows())
                for column in range(self.num_columns()))

    def evaluate(self, index_iterator: Iterator[Tuple[int, int]]) -> Iterator[Tuple[int, int, real]]:
        """
        Takes an iterator over matrix indices and includes the value at each index.

        :param index_iterator:  The iterator over (row, column) pairs.
        :return:                An iterator over (row, column, value) triplets.
        """
        return ((row, column, self._data[row][column])
                for row, column in index_iterator)

    def evaluate_only(self, index_iterator: Iterator[Tuple[int, int]]) -> Iterator[real]:
        """
        Takes an iterator over matrix indices and returns an iterator over the
        values at those indices.

        :param index_iterator:  The iterator over (row, column) pairs.
        :return:                An iterator over matrix values.
        """
        return (value for row, column, value in self.evaluate(index_iterator))

    # ------------- #
    # SERIALISATION #
    # ------------- #

    def serialise_state(self, stream: IO[bytes]):
        # Write the number of rows and columns
        stream.write(self.serialise_to_bytes(self.num_rows()))
        stream.write(self.serialise_to_bytes(self.num_columns()))

        # Write each data element in order
        for row_index in range(self.num_rows()):
            for column_index in range(self.num_columns()):
                stream.write(self.serialise_to_bytes(self.get(row_index, column_index)))

    # ---------------------- #
    # STRING REPRESENTATIONS #
    # ---------------------- #

    def __str__(self):
        return "\n".join(self.row_str(index) for index in range(self.num_rows()))

    def row_str(self, row_index: int) -> str:
        """
        Gets the string representation of a single row of this matrix.

        :param row_index:   The row to get the string representation for.
        :return:            The string representation.
        """
        return ",".join(str(self._data[row_index][column_index])
                        for column_index in range(self.num_columns()))

    def shape_string(self) -> str:
        """
        Gets the string representation of the shape of this matrix.

        :return:    The shape's string representation.
        """
        return f"[{self.num_rows()} x {self.num_columns()}]"

    # ----- #
    # OTHER #
    # ----- #

    def transpose(self) -> 'Matrix':
        """
        Returns the matrix transpose of this matrix.

        :return:    The transposed matrix.
        """
        return Matrix(self._data.transpose().copy())

    # Alias for transpose
    t = transpose

    def copy(self) -> 'Matrix':
        """
        Returns a copy of this matrix.

        :return:    An identical copy.
        """
        return Matrix(self._data.copy())

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

    def vector_to_diagonal(self):
        """
        TODO: Comment

        :return:
        """
        self.ensure_is_vector()
        raw = [[self.get_flat(i) if i == j else real(0)
                for i in range(self.num_elements())]
               for j in range(self.num_elements())]
        return Matrix(np.array(raw))

    # --------- #
    # _INTERNAL #
    # --------- #

    def _reset_cache(self):
        """
        Resets the cached decompositions when a matrix value is changed so that
        they are recalculated.
        """
        self._eigenvalue_decomposition = None
        self._singular_value_decomposition = None
        self._qr_decomposition = None

    def __eq__(self, other):
        # Always equal to ourselves
        if self is other:
            return True

        # Can only be equal to other matrices
        if not isinstance(other, Matrix):
            return False

        # Must be that same shape as each other
        if not self.is_same_shape_as(other):
            return False

        # Calculate the difference between the matrices
        diff: Matrix = self.subtract(other)

        return not diff._data.any()

    def __hash__(self):
        # TODO
        raise NotImplementedError
