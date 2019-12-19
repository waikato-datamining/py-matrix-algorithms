#  _GLSW.py
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

from ...core.algorithm import SupervisedMatrixAlgorithm
from ...core.errors import MatrixAlgorithmsError
from ...core.matrix import Matrix, factory
from .._Center import Center


class GLSW(SupervisedMatrixAlgorithm):
    def __init__(self):
        super().__init__()

        self._alpha: float = 1e-3
        self._G: Optional[Matrix] = None

    def get_alpha(self) -> float:
        return self._alpha

    def set_alpha(self, value: float):
        if value <= 0:
            raise ValueError(f"alpha must be greater than zero, got {value}")

        self._alpha = value
        self.reset()

    alpha = property(get_alpha, set_alpha)

    def get_projection_matrix(self) -> Matrix:
        return self._G

    G = property(get_projection_matrix)

    def _do_reset(self):
        super()._do_reset()

        self._G = None

    def to_string(self) -> str:
        if self.is_configured():
            return 'Generalized Least Squares Weighting. Projection Matrix shape: ' + self._G.shape_string()
        else:
            return 'Generalized Least Squares Weighting. Model not yet initialized.'

    def _check(self, X: Matrix, y: Matrix):
        if not X.is_same_shape_as(y):
            raise MatrixAlgorithmsError("Matrices X and y must have the same shape")

    def _do_configure(self, X: Matrix, y: Matrix):
        self._check(X, y)

        C: Matrix = self.get_covariance_matrix(X, y)

        # SVD
        V: Matrix = self.get_eigenvector_matrix(C)
        D: Matrix = self.get_weight_matrix(C)

        # Projection matrix
        self._G = V.matrix_multiply(D.inverse()).matrix_multiply(V.transpose())

    def get_eigenvector_matrix(self, C: Matrix) -> Matrix:
        return C.get_eigenvalue_decomposition_V()

    def get_weight_matrix(self, C: Matrix) -> Matrix:
        # Get eigenvalues
        S_squared: Matrix = C.svd_S().pow(2)

        # Weights
        D: Matrix = S_squared.divide(self._alpha)
        D = D.add(factory.eye_like(D))
        D = D.sqrt()
        return D

    def get_covariance_matrix(self, x1: Matrix, x2: Matrix) -> Matrix:
        # Center X1, X2
        c1: Center = Center()
        c2: Center = Center()
        x1_centered: Matrix = c1.configure_and_transform(x1)
        x2_centered: Matrix = c2.configure_and_transform(x2)

        # Build difference
        X_d: Matrix = x2_centered.subtract(x1_centered)

        # Covariance Matrix
        return X_d.transpose().matrix_multiply(X_d)

    def _do_transform(self, predictors: Matrix) -> Matrix:
        return predictors.matrix_multiply(self._G)

    def is_non_invertible(self) -> bool:
        return True
