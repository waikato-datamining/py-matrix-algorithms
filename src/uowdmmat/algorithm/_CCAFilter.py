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

from ..core import SupervisedFilter
from ._AbstractAlgorithm import AbstractAlgorithm
from ..core import real
from ..core.matrix import Matrix, factory
from ..transformation import Center


class CCAFilter(AbstractAlgorithm, SupervisedFilter):
    def __init__(self):
        super().__init__()
        self.lambda_X: real = real(1e-2)
        self.lambda_Y: real = real(1e-2)
        self.kcca: int = 1
        self.center_X: Optional[Center] = None
        self.center_Y: Optional[Center] = None
        self.proj_X: Optional[Matrix] = None
        self.proj_Y: Optional[Matrix] = None

    @staticmethod
    def validate_kcca(value: int):
        return value > 0

    @staticmethod
    def validate_lambda_X(value: real):
        return True

    @staticmethod
    def validate_lambda_Y(value: real):
        return True

    def to_string(self) -> str:
        return 'Canonical Correlation Analysis Filter (CCARegression)'

    def reset(self):
        super().reset()
        self.proj_X = None
        self.proj_Y = None

    def initialize(self, x: Optional[Matrix] = None, y: Optional[Matrix] = None) -> Optional[str]:
        if x is None or y is None:
            super().initialize()
            self.kcca = 1
            self.lambda_X = real(1e-2)
            self.lambda_Y = real(1e-2)
            self.center_X = Center()
            self.center_Y = Center()
        else:
            x = x.copy()
            y = y.copy()

            self.reset()

            result = self.check(x, y)

            if result is None:
                result = self.do_initialize(x, y)
                self.initialised = (result is None)

            return result

    def do_initialize(self, X: Matrix, Y: Matrix) -> Optional[str]:
        super().initialize()

        num_features: int = X.num_columns()
        num_targets: int = Y.num_columns()

        # Check if dimension kcca is valid
        if self.kcca > min(num_features, num_targets):
            return 'Projection dimension must be <= min(X.num_columns(), Y.num_columns()).'

        # Center input
        X = self.center_X.transform(X)
        Y = self.center_Y.transform(Y)

        # Regularisation matrices
        lambda_IX: Matrix = factory.eye(num_features).mul(self.lambda_X)
        lambda_IY: Matrix = factory.eye(num_targets).mul(self.lambda_Y)

        # Get covariance matrices
        Cxx: Matrix = X.transpose().mul(X).add(lambda_IX)
        Cyy: Matrix = Y.transpose().mul(Y).add(lambda_IY)
        Cxy: Matrix = X.transpose().mul(Y)

        # Apply A^(-1/2)
        Cxx_inv_sqrt: Matrix = self.pow_minus_half(Cxx)
        Cyy_inv_sqrt: Matrix = self.pow_minus_half(Cyy)

        # Calculate omega for SVD
        omega: Matrix = Cxx_inv_sqrt.mul(Cxy).mul(Cyy_inv_sqrt)

        U: Matrix = omega.svd_U().normalized(0)
        V: Matrix = omega.svd_V().normalized(0)

        C: Matrix = U.get_sub_matrix((0, U.num_rows()), (0, self.kcca))
        D: Matrix = V.get_sub_matrix((0, V.num_rows()), (0, self.kcca))

        self.proj_X = Cxx_inv_sqrt.mul(C)
        self.proj_Y = Cyy_inv_sqrt.mul(D)

        return None

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
        A_pow_half: Matrix = eig_vecs_desc.mul(d_sqrt_inv).mul(eig_vecs_desc.transpose())
        return A_pow_half

    def check(self, x: Optional[Matrix], y: Optional[Matrix]) -> Optional[str]:
        """
        Hook method for checking the data before training.

        :param x:   First sample set.
        :param y:   Second sample set.
        :return:    None if successful, otherwise error message.
        """
        if x is None:
            return 'No x matrix provided!'
        if y is None:
            return 'No y matrix provided!'
        return None

    def do_transform(self, predictors: Matrix) -> Matrix:
        """
        Transforms the predictors data.

        :param predictors:  The input data.
        :return:            The transformed data.
        """
        predictors = self.center_X.transform(predictors)
        return predictors.mul(self.proj_X)

    def do_transform_response(self, response: Matrix) -> Matrix:
        """
        Transforms the response data.

        :param response:    The input data.
        :return:            The transformed data.
        """
        response = self.center_Y.transform(response)
        return response.mul(self.proj_Y)

    def transform(self, predictors: Matrix) -> Matrix:
        if not self.is_initialised():
            raise RuntimeError('Algorithm has not been initialised!')

        return self.do_transform(predictors)

    def transform_response(self, response: Matrix) -> Matrix:
        if not self.is_initialised():
            raise RuntimeError('Algorithm has not been initialised!')

        return self.do_transform_response(response)


