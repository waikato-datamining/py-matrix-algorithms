#  _CCAFilter.py
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
from typing import Optional

from ..core.algorithm import SupervisedMatrixAlgorithmWithResponseTransform
from ..core import real
from ..core.matrix import Matrix, factory, Axis
from ._Center import Center


class CCAFilter(SupervisedMatrixAlgorithmWithResponseTransform):
    def __init__(self):
        super().__init__()

        # Defaults
        self._lambda_X: real = real(1e-2)
        self._lambda_Y: real = real(1e-2)
        self._kcca: int = 1

        self._center_X: Center = Center()
        self._center_Y: Center = Center()
        self._proj_X: Optional[Matrix] = None
        self._proj_Y: Optional[Matrix] = None

    def get_lambda_X(self) -> real:
        return self._lambda_X

    def set_lambda_X(self, value: real):
        self._lambda_X = value
        self.reset()

    lambda_X = property(get_lambda_X, set_lambda_X)

    def get_lambda_Y(self) -> real:
        return self._lambda_Y

    def set_lambda_Y(self, value: real):
        self._lambda_Y = value
        self.reset()

    lambda_Y = property(get_lambda_Y, set_lambda_Y)

    def get_kcca(self) -> int:
        return self._kcca

    def set_kcca(self, value: int):
        if value < 1:
            raise ValueError(f"Target dimension kcca must be at least 1 but was {value}")

        self._kcca = value
        self.reset()

    kcca = property(get_kcca, set_kcca)

    def get_projection_matrix_X(self) -> Matrix:
        return self._proj_X

    def get_projection_matrix_Y(self) -> Matrix:
        return self._proj_Y

    def __str__(self):
        return "Canonical Correlation Analysis Filter (CCARegression)"

    def _do_reset(self):
        super()._do_reset()

        self._proj_X = None
        self._proj_Y = None
        self._center_X.reset()
        self._center_Y.reset()

    def _do_configure(self, X: Matrix, y: Matrix):
        num_features: int = X.num_columns()
        num_targets: int = y.num_columns()

        # Check if dimension kcca is valid
        if self._kcca > min(num_features, num_targets):
            raise ValueError(f"Projection dimension cannot be greater than the number of columns"
                             f"in either X or y ({num_features} and {num_targets}) but is {self._kcca}")

        # Center input
        X = self._center_X.configure_and_transform(X)
        y = self._center_Y.configure_and_transform(y)

        # Regularisation matrices
        lambda_IX: Matrix = factory.eye(num_features).multiply(self._lambda_X)
        lambda_IY: Matrix = factory.eye(num_targets).multiply(self._lambda_Y)

        # Get covariance matrices
        Cxx: Matrix = X.transpose().matrix_multiply(X).add(lambda_IX)
        Cyy: Matrix = y.transpose().matrix_multiply(y).add(lambda_IY)
        Cxy: Matrix = X.transpose().matrix_multiply(y)

        # Apply A^(-1/2)
        Cxx_inv_sqrt: Matrix = self.pow_minus_half(Cxx)
        Cyy_inv_sqrt: Matrix = self.pow_minus_half(Cyy)

        # Calculate omega for SVD
        omega: Matrix = Cxx_inv_sqrt.matrix_multiply(Cxy).matrix_multiply(Cyy_inv_sqrt)

        U: Matrix = omega.svd_U().normalized(Axis.COLUMNS)
        V: Matrix = omega.svd_V().normalized(Axis.COLUMNS)

        C: Matrix = U.get_sub_matrix((0, U.num_rows()), (0, self._kcca))
        D: Matrix = V.get_sub_matrix((0, V.num_rows()), (0, self._kcca))

        self._proj_X = Cxx_inv_sqrt.matrix_multiply(C)
        self._proj_Y = Cyy_inv_sqrt.matrix_multiply(D)

    def pow_minus_half(self, A: Matrix) -> Matrix:
        """
        Compute A^(-1/2) = (A^(-1))^(1/2) on a matrix A, where A^(1/2) = M
        with A = MM.

        :param A:   Input matrix.
        :return:    A^(-1/2).
        """
        eig_vals_desc: Matrix = A.get_eigenvalues_sorted_descending()
        eig_vecs_desc: Matrix = A.get_eigenvectors(True)
        diag: Matrix = factory.diag(eig_vals_desc)
        d_sqrt_inv: Matrix = diag.sqrt().inverse()
        A_pow_half: Matrix = eig_vecs_desc.matrix_multiply(d_sqrt_inv).matrix_multiply(eig_vecs_desc.transpose())
        return A_pow_half

    def _do_transform(self, predictors: Matrix) -> Matrix:
        """
        Transforms the predictors data.

        :param predictors:  The input data.
        :return:            The transformed data.
        """
        predictors = self._center_X.transform(predictors)
        return predictors.matrix_multiply(self._proj_X)

    def _do_transform_response(self, response: Matrix) -> Matrix:
        """
        Transforms the response data.

        :param response:    The input data.
        :return:            The transformed data.
        """
        response = self._center_Y.transform(response)
        return response.matrix_multiply(self._proj_Y)

    def is_non_invertible(self) -> bool:
        return True
