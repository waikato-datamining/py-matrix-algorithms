#  _SparsePLS.py
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
from typing import Optional, Set, List

from wai.common import switch, case, default

from ._AbstractSingleResponsePLS import AbstractSingleResponsePLS
from ._NIPALS import NIPALS
from ...core import ZERO, real
from ...core.matrix import Matrix, factory
from .._Standardize import Standardize


class SparsePLS(AbstractSingleResponsePLS):
    """
    Sparse PLS algorithm.

    See here:
    <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2810828/">
        Sparse partial least squares regression for simultaneous dimension reduction and variable selection
    </a>

    Implementation was oriented at the R SPLS package, which implements the above
    mentioned paper:
    <a href="https://github.com/cran/spls">
        Sparse Partial Least Squares (SPLS) Regression and Classification
    </a>

    The lambda parameter controls the features sparseness. For sufficiently small
    lambda, all features will be selected and the algorithm results are equal
    to NIPALS'.
    """
    def __init__(self):
        super().__init__()
        self._B_pls: Optional[Matrix] = None
        self._tol: real = real(1e-7)  # NIPALS tolerance threshold
        self._max_iter: int = 500  # NIPALS max iterations
        self._lambda: real = real(0.5)  # Sparsity parameter. Determines sparseness
        self._A: Optional[Set[int]] = None
        self._W: Optional[Matrix] = None  # Loadings
        self._standardize_X: Standardize = Standardize()  # Standardize X
        self._standardize_Y: Standardize = Standardize()  # Standardize Y

    def get_max_iter(self) -> int:
        return self._max_iter

    def set_max_iter(self, value: int):
        if value < 0:
            raise ValueError(f"Maximum iteration parameter must be positive but was {value}")

        self._max_iter = value
        self.reset()

    max_iter = property(get_max_iter, set_max_iter)

    def get_tol(self) -> real:
        return self._tol

    def set_tol(self, value: real):
        if value < 0:
            raise ValueError(f"Tolerance parameter must be positive but was {value}")

        self._tol = value
        self.reset()

    tol = property(get_tol, set_tol)

    def get_lambda(self) -> real:
        return self._lambda

    def set_lambda(self, value: real):
        if abs(value) < 0:
            raise ValueError(f"Sparseness parameter lambda must be postive but was {value}")

        self._lambda = value
        self.reset()

    lambda_ = property(get_lambda, set_lambda)

    def _do_reset(self):
        """
        Resets the member variables.
        """
        super()._do_reset()
        self._B_pls = None
        self._A = None
        self._W = None
        self._standardize_X.reset()
        self._standardize_Y.reset()

    def get_matrix_names(self) -> List[str]:
        """
        Returns the names of all the available matrices.

        :return:    The names of the matrices.
        """
        return ['W',
                'B']

    def get_matrix(self, name: str) -> Optional[Matrix]:
        """
        Returns the matrix with the specified name.

        :param name:    The name of the matrix.
        :return:        The matrix, None if not available.
        """
        with switch(name):
            if case('W'):
                return self._W
            if case('B'):
                return self._B_pls
            if default():
                return None

    def has_loadings(self) -> bool:
        """
        Whether the algorithm supports the return of loadings.

        :return:    True if supported.
        """
        return True

    def get_loadings(self) -> Optional[Matrix]:
        """
        Returns the loadings, if available.

        :return:    The loadings, None if not available.
        """
        return self._W

    def _do_pls_configure(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        """
        Initialises using the provided data.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        X: Matrix = self._standardize_X.configure_and_transform(predictors)
        y: Matrix = self._standardize_Y.configure_and_transform(response)
        X_j: Matrix = X.copy()
        y_j: Matrix = y.copy()
        self._A = set()
        self._B_pls = factory.zeros(X.num_columns(), y.num_columns())
        self._W = factory.zeros(X.num_columns(), self._num_components)

        for k in range(self._num_components):
            w_k: Matrix = self.get_direction_vector(X_j, y_j, k)
            self._W.set_column(k, w_k)

            if self.debug:
                self.check_direction_vector(w_k)

            self.collect_indices(w_k)

            X_A: Matrix = self.get_column_submatrix_of(X)
            self._B_pls = factory.zeros(X.num_columns(), y.num_columns())
            B_pls_A: Matrix = self.get_regression_coefficient(X_A, y, k)

            # Fill self.B_pls values at non zero indices with estimated
            # regression coefficients
            idx_counter: int = 0
            for idx in sorted(self._A):
                self._B_pls.set_row(idx, B_pls_A.get_row(idx_counter))
                idx_counter += 1

            # Deflate
            y_j = y.subtract(X.matrix_multiply(self._B_pls))

        if self.debug:
            self.logger.info('Selected following features ' +
                             '(' + str(len(self._A)) + '/' + str(X.num_columns()) + '): ')
            l: List[str] = [str(a) for a in sorted(self._A)]
            self.logger.info(','.join(l))

        return None

    def get_regression_coefficient(self, X_A: Matrix, y: Matrix, k: int) -> Matrix:
        """
        Calculate NIPALS regression coefficient.

        :param X_A:     Predictors subset.
        :param y:       Current response vector.
        :param k:       PLS iteration.
        :return:        B_pls (NIPALS regression coefficients)
        """
        num_components: int = min(X_A.num_columns(), k + 1)
        nipals: NIPALS = NIPALS()
        nipals._max_iter = self._max_iter
        nipals._tol = self._tol
        nipals._num_components = num_components
        nipals.configure(X_A, y)
        return nipals._coef

    def get_column_submatrix_of(self, X: Matrix) -> Matrix:
        """
        Get the column submatrix of X given by the indices in self.A.

        :param X:   Input matrix.
        :return:    Submatrix of X.
        """
        X_A: Matrix = factory.zeros(X.num_rows(), len(self._A))
        col_count: int = 0
        for i in sorted(self._A):
            col: Matrix = X.get_column(i)
            X_A.set_column(col_count, col)
            col_count += 1
        return X_A

    def get_row_submatrix_of(self, X: Matrix) -> Matrix:
        """
        Get the row submatrix of X given by the indices in self.A.

        :param X:   Input matrix.
        :return:    Submatrix of X.
        """
        X_A: Matrix = factory.zeros(len(self._A), X.num_columns())
        row_count: int = 0
        for i in sorted(self._A):
            row: Matrix = X.get_row(i)
            X_A.set_row(row_count, row)
            row_count += 1
        return X_A

    def collect_indices(self, w: Matrix):
        """
        Collect indices based on the current non zero indices in w and self.B_pls.

        :param w:   Direction vector.
        """
        self._A.clear()
        self._A.update(w.where_vector(lambda d: abs(d) > 1e-6))
        self._A.update(self._B_pls.where_vector(lambda d: abs(d) > 1e-6))

    def check_direction_vector(self, w: Matrix):
        """
        Check if the direction vector fulfills w^Tw=1.

        :param w:   Direction vector.
        """
        # Test if w^Tw = 1
        if w.norm2_squared() - 1 > 1e-6:
            self.logger.warning("Direction vector condition w'w=1 was violated.")

    def get_direction_vector(self, X: Matrix, y_j: Matrix, k: int) -> Matrix:
        """
        Compute the direction vector.

        :param X:       Predictors.
        :param y_j:     Current deflated response.
        :param k:       Iteration.
        :return:        Direction vector.
        """
        Z_p: Matrix = X.transpose().matrix_multiply(y_j)
        z_norm: real = Z_p.abs().median().as_scalar()  # R package spls uses median norm
        Z_p = Z_p.divide(z_norm)
        Z_P_sign: Matrix = Z_p.sign()
        val_b: Matrix = Z_p.abs().subtract(self._lambda * Z_p.abs().maximum().as_scalar())

        # Collect indices where val_b is >= 0
        idxs: List[int] = val_b.where_vector(lambda d: d >= 0)
        pre_mul: Matrix = val_b.multiply(Z_P_sign)
        c: Matrix = factory.zeros(Z_p.num_rows(), 1)
        for idx in idxs:
            val: real = pre_mul.get(idx, 0)
            c.set(idx, 0, val)

        return c.divide(c.norm2_squared())  # Rescale c and use as estimated direction vector

    def _do_pls_transform(self, predictors: Matrix) -> Matrix:
        """
        Transforms the data.

        :param predictors:  The input data.
        :return:            The transformed data and the predictions.
        """
        num_components: int = self._num_components
        T: Matrix = factory.zeros(predictors.num_rows(), num_components)
        X: Matrix = predictors.copy()
        for k in range(num_components):
            w_k: Matrix = self._W.get_column(k)
            t_k: Matrix = X.matrix_multiply(w_k)
            T.set_column(k, t_k)

            p_k: Matrix = X.transpose().matrix_multiply(t_k).divide(t_k.norm2_squared())
            X = X.subtract(t_k.matrix_multiply(p_k.transpose()))

        return T

    def can_predict(self) -> bool:
        """
        Returns whether the algorithm can make predictions.

        :return:    True if can make predictions.
        """
        return True

    def _do_pls_predict(self, predictors: Matrix) -> Matrix:
        """
        Performs predictions on the data.

        :param predictors:  The input data.
        :return:            The transformed data and the predictions.
        """
        X: Matrix = self._standardize_X.transform(predictors)

        X_A: Matrix = self.get_column_submatrix_of(X)
        B_A: Matrix = self.get_row_submatrix_of(self._B_pls)

        y_means: Matrix = self._standardize_Y.get_means()
        y_std: Matrix = self._standardize_Y.get_std_devs()
        y_hat: Matrix = X_A.matrix_multiply(B_A).multiply(y_std).add(y_means)

        return y_hat
