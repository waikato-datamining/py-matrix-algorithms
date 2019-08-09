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
from ...transformation import Standardize


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
        self.B_pls: Optional[Matrix] = None
        self.tol: real = ZERO  # NIPALS tolerance threshold
        self.max_iter: int = 0  # NIPALS max iterations
        self.lambda_: real = ZERO  # Sparsity parameter. Determines sparseness
        self.A: Optional[Set[int]] = None
        self.W: Optional[Matrix] = None  # Loadings
        self.standardize_X: Optional[Standardize] = None  # Standardize X
        self.standardize_Y: Optional[Standardize] = None  # Standardize Y

    @staticmethod
    def validate_max_iter(value: int) -> bool:
        return value >= 0

    @staticmethod
    def validate_tol(value: real) -> bool:
        return value >= 0

    @staticmethod
    def validate_lambda_(value: real) -> bool:
        return value >= 0

    def reset(self):
        """
        Resets the member variables.
        """
        super().reset()
        self.B_pls = None
        self.A = None
        self.W = None
        self.standardize_X = Standardize()
        self.standardize_Y = Standardize()

    def initialize(self, predictors: Optional[Matrix] = None, response: Optional[Matrix] = None) -> Optional[str]:
        if predictors is None and response is None:
            super().initialize()
            self.lambda_ = real(0.5)
            self.tol = real(1e-7)
            self.max_iter = 500
            self.standardize_X = Standardize()
            self.standardize_Y = Standardize()
        else:
            return super().initialize(predictors, response)

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
                return self.W
            if case('B'):
                return self.B_pls
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
        return self.W

    def do_perform_initialization(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        """
        Initialises using the provided data.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        X: Matrix = self.standardize_X.transform(predictors)
        y: Matrix = self.standardize_Y.transform(response)
        X_j: Matrix = X.copy()
        y_j: Matrix = y.copy()
        self.A = set()
        self.B_pls = factory.zeros(X.num_columns(), y.num_columns())
        self.W = factory.zeros(X.num_columns(), self.num_components)

        for k in range(self.num_components):
            w_k: Matrix = self.get_direction_vector(X_j, y_j, k)
            self.W.set_column(k, w_k)

            if self.debug:
                self.check_direction_vector(w_k)

            self.collect_indices(w_k)

            X_A: Matrix = self.get_column_submatrix_of(X)
            self.B_pls = factory.zeros(X.num_columns(), y.num_columns())
            B_pls_A: Matrix = self.get_regression_coefficient(X_A, y, k)

            # Fill self.B_pls values at non zero indices with estimated
            # regression coefficients
            idx_counter: int = 0
            for idx in sorted(self.A):
                self.B_pls.set_row(idx, B_pls_A.get_row(idx_counter))
                idx_counter += 1

            # Deflate
            y_j = y.sub(X.mul(self.B_pls))

        if self.debug:
            self.logger.info('Selected following features ' +
                             '(' + str(len(self.A)) + '/' + str(X.num_columns()) + '): ')
            l: List[str] = [str(a) for a in sorted(self.A)]
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
        nipals.max_iter = self.max_iter
        nipals.tol = self.tol
        nipals.num_components = num_components
        nipals.initialize(X_A, y)
        return nipals.coef

    def get_column_submatrix_of(self, X: Matrix) -> Matrix:
        """
        Get the column submatrix of X given by the indices in self.A.

        :param X:   Input matrix.
        :return:    Submatrix of X.
        """
        X_A: Matrix = factory.zeros(X.num_rows(), len(self.A))
        col_count: int = 0
        for i in sorted(self.A):
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
        X_A: Matrix = factory.zeros(len(self.A), X.num_columns())
        row_count: int = 0
        for i in sorted(self.A):
            row: Matrix = X.get_row(i)
            X_A.set_row(row_count, row)
            row_count += 1
        return X_A

    def collect_indices(self, w: Matrix):
        """
        Collect indices based on the current non zero indices in w and self.B_pls.

        :param w:   Direction vector.
        """
        self.A.clear()
        self.A.update(w.where_vector(lambda d: abs(d) > 1e-6))
        self.A.update(self.B_pls.where_vector(lambda d: abs(d) > 1e-6))

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
        Z_p: Matrix = X.transpose().mul(y_j)
        z_norm: real = Z_p.abs().median()  # R package spls uses median norm
        Z_p = Z_p.div(z_norm)
        Z_P_sign: Matrix = Z_p.sign()
        val_b: Matrix = Z_p.abs().sub(self.lambda_ * Z_p.abs().max())

        # Collect indices where val_b is >= 0
        idxs: List[int] = val_b.where_vector(lambda d: d >= 0)
        pre_mul: Matrix = val_b.mul_elementwise(Z_P_sign)
        c: Matrix = factory.zeros(Z_p.num_rows(), 1)
        for idx in idxs:
            val: real = pre_mul.get(idx, 0)
            c.set(idx, 0, val)

        return c.div(c.norm2_squared())  # Rescale c and use as estimated direction vector

    def do_transform(self, predictors: Matrix) -> Matrix:
        """
        Transforms the data.

        :param predictors:  The input data.
        :return:            The transformed data and the predictions.
        """
        num_components: int = self.num_components
        T: Matrix = factory.zeros(predictors.num_rows(), num_components)
        X: Matrix = predictors.copy()
        for k in range(num_components):
            w_k: Matrix = self.W.get_column(k)
            t_k: Matrix = X.mul(w_k)
            T.set_column(k, t_k)

            p_k: Matrix = X.transpose().mul(t_k).div(t_k.norm2_squared())
            X = X.sub(t_k.mul(p_k.transpose()))

        return T

    def can_predict(self) -> bool:
        """
        Returns whether the algorithm can make predictions.

        :return:    True if can make predictions.
        """
        return True

    def do_perform_predictions(self, predictors: Matrix) -> Matrix:
        """
        Performs predictions on the data.

        :param predictors:  The input data.
        :return:            The transformed data and the predictions.
        """
        X: Matrix = self.standardize_X.transform(predictors)

        X_A: Matrix = self.get_column_submatrix_of(X)
        B_A: Matrix = self.get_row_submatrix_of(self.B_pls)

        y_means: Matrix = self.standardize_Y.get_means()
        y_std: Matrix = self.standardize_Y.get_std_devs()
        y_hat: Matrix = X_A.mul(B_A).scale_by_row_vector(y_std).add_by_vector(y_means)

        return y_hat
