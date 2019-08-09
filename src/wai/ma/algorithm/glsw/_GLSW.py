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

from ...core import SupervisedFilter
from .._AbstractAlgorithm import AbstractAlgorithm
from ...core.matrix import Matrix, factory
from ...transformation import Center


class GLSW(AbstractAlgorithm, SupervisedFilter):
    def __init__(self):
        super().__init__()
        self.alpha: float = 1e-3
        self.G: Optional[Matrix] = None

    @staticmethod
    def validate_alpha(value: float) -> bool:
        return value > 0

    def reset(self):
        super().reset()
        self.G = None

    def initialize(self, x1: Optional[Matrix] = None, x2: Optional[Matrix] = None) -> Optional[str]:
        if x1 is None and x2 is None:
            super().initialize()
            self.alpha = 1e-3
        else:
            # Always work on copies
            x1 = x1.copy()
            x2 = x2.copy()

            self.reset()

            result: Optional[str] = self.check(x1, x2)

            if result is None:
                result = self.do_initialize(x1, x2)
                self.initialised = result is None

            return result

    def to_string(self) -> str:
        if self.initialised:
            return 'Generalized Least Squares Weighting. Projection Matrix shape: ' + self.G.shape_string()
        else:
            return 'Generalized Least Squares Weighting. Model not yet initialized.'

    def do_initialize(self, x1: Matrix, x2: Matrix) -> Optional[str]:
        super().initialize()

        C: Matrix = self.get_covariance_matrix(x1, x2)

        # SVD
        V: Matrix = self.get_eigenvector_matrix(C)
        D: Matrix = self.get_weight_matrix(C)

        # Projection matrix
        self.G = V.mul(D.inverse()).mul(V.transpose())

        return None

    def get_eigenvector_matrix(self, C: Matrix) -> Matrix:
        return C.get_eigenvalue_decomposition_V()

    def get_weight_matrix(self, C: Matrix) -> Matrix:
        # Get eigenvalues
        S_squared: Matrix = C.svd_S().pow_elementwise(2)

        # Weights
        D: Matrix = S_squared.div(self.alpha)
        D = D.add(factory.eye_like(D))
        D = D.sqrt()
        return D

    def get_covariance_matrix(self, x1: Matrix, x2: Matrix) -> Matrix:
        # Center X1, X2
        c1: Center = Center()
        c2: Center = Center()
        x1_centered: Matrix = c1.transform(x1)
        x2_centered: Matrix = c2.transform(x2)

        # Build difference
        X_d: Matrix = x2_centered.sub(x1_centered)

        # Covariance Matrix
        return X_d.transpose().mul(X_d)

    def do_transform(self, predictors: Matrix) -> Matrix:
        return predictors.mul(self.G)

    def transform(self, predictors: Matrix) -> Matrix:
        if not self.is_initialised():
            raise RuntimeError("Algorithm hasn't been initialized!")

        return self.do_transform(predictors)

    def check(self, x1: Matrix, x2: Matrix) -> Optional[str]:
        if x1 is None:
            return 'No x1 matrix provided!'
        if x2 is None:
            return 'No x2 matrix provided!'
        if not x1.same_shape_as(x2):
            return 'Matrices x1 and x2 must have the same shape'
        return None
