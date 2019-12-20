#  _NIPALS.py
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
from enum import Enum, auto
from typing import Optional, List

from ._AbstractMultiResponsePLS import AbstractMultiResponsePLS
from ...core import real, ONE
from ...core.matrix import Matrix, factory
from .._Standardize import Standardize


class NIPALS(AbstractMultiResponsePLS):
    """
    Nonlinear Iterative Partial Least Squares

    Implementation oriented at scikit-learn's NIPALS implementation:
    <a href="https://github.com/scikit-learn/scikit-learn/blob/ed5e127b/sklearn/cross_decomposition/pls_.py#L455">
        Github scikit-learn NIPALS
    </a>

    Parameters:
    - tol: Iterative convergence tolerance
    - maxIter: Maximum number of iterations
    - normYWeights: Flat to normalize Y weights
    - deflationMode: Mode for Y matrix deflation. Can be either CANONICAL or REGRESSION
    """
    def __init__(self):
        super().__init__()
        self._X_scores: Optional[Matrix] = None  # Scores on X
        self._Y_scores: Optional[Matrix] = None  # Scores on Y
        self._X_loadings: Optional[Matrix] = None  # Loadings on X
        self._Y_loadings: Optional[Matrix] = None  # Loadings on Y
        self._X_weights: Optional[Matrix] = None  # Weights on X
        self._Y_weights: Optional[Matrix] = None  # Weights on Y
        self._X_rotations: Optional[Matrix] = None  # Projection of X into latent space
        self._Y_rotations: Optional[Matrix] = None  # Projection of Y into latent space
        self._X: Optional[Matrix] = None  # Training points
        self._coef: Optional[Matrix] = None  # Regression coefficients
        self._tol: real = real(1e-6)  # Inner NIPALS loop improvement tolerance
        self._max_iter: int = 500  # Inner NIPALS loop maximum number of iterations
        self.norm_Y_weights: bool = False  # Flag to normalize Y weights
        self._standardize_X: Standardize = Standardize()  # Standarize X tranformation
        self._standardize_Y: Standardize = Standardize()  # Standardize Y transformation
        self._deflation_mode: DeflationMode = DeflationMode.REGRESSION

    def is_norm_Y_weights(self) -> bool:
        return self.norm_Y_weights

    def set_norm_Y_weights(self, value: bool):
        self.norm_Y_weights = value

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

    def get_deflation_mode(self) -> "DeflationMode":
        return self._deflation_mode

    def set_deflation_mode(self, value: "DeflationMode"):
        self._deflation_mode = value

    deflation_mode = property(get_deflation_mode, set_deflation_mode)

    def get_min_columns_response(self) -> int:
        return 1

    def get_max_columns_response(self) -> int:
        return -1

    def _do_pls_configure(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        # Init
        X: Matrix = predictors
        X = self._standardize_X.configure_and_transform(X)
        Y: Matrix = response
        Y = self._standardize_Y.configure_and_transform(Y)

        # Dimensions
        num_rows: int = X.num_rows()
        num_features: int = X.num_columns()
        num_classes: int = Y.num_columns()
        num_components: int = self._num_components

        # Init matrices
        self._X_scores = factory.zeros(num_rows, num_components)  # T
        self._Y_scores = factory.zeros(num_rows, num_components)  # U

        self._X_weights = factory.zeros(num_features, num_components)  # W
        self._Y_weights = factory.zeros(num_classes, num_components)  # C

        self._X_loadings = factory.zeros(num_features, num_components)  # P
        self._Y_loadings = factory.zeros(num_classes, num_components)  # Q

        yk_loading: Matrix = factory.zeros(num_classes, 1)

        eps: real = real(1e-10)
        for k in range(num_components):
            if Y.transpose().matrix_multiply(Y).all(lambda e: e < eps):
                self.logger.warning('Y residual constant at iteration ' + str(k))
                break

            res: NipalsLoopResult = self.nipals_loop(X, Y)
            xk_weight: Matrix = res.X_weights
            yk_weight: Matrix = res.Y_weights

            # Calculate latent X and Y scores
            xk_score: Matrix = X.matrix_multiply(xk_weight)
            yk_score: Matrix = Y.matrix_multiply(yk_weight).divide(yk_weight.norm2_squared())

            if xk_score.norm2_squared() < eps:
                self.logger.warning('X scores are null at component ' + str(k))
                break

            # Deflate X
            xk_loading: Matrix = X.transpose().matrix_multiply(xk_score).divide(xk_score.norm2_squared())
            X = X.subtract(xk_score.matrix_multiply(xk_loading.transpose()))

            # Deflate Y
            if self.deflation_mode is DeflationMode.CANONICAL:
                yk_loading: Matrix = Y.transpose().matrix_multiply(yk_score).divide(yk_score.norm2_squared())
                Y = Y.subtract(yk_score.matrix_multiply(yk_loading.transpose()))
            elif self.deflation_mode is DeflationMode.REGRESSION:
                yk_loading: Matrix = Y.transpose().matrix_multiply(xk_score).divide(xk_score.norm2_squared())
                Y = Y.subtract(xk_score.matrix_multiply(yk_loading.transpose()))

            # Store results
            self._X_scores.set_column(k, xk_score)
            self._Y_scores.set_column(k, yk_score)
            self._X_weights.set_column(k, xk_weight)
            self._Y_weights.set_column(k, yk_weight)
            self._X_loadings.set_column(k, xk_loading)
            self._Y_loadings.set_column(k, yk_loading)

        self._X = X
        self._X_rotations = self._X_weights.matrix_multiply((self._X_loadings.transpose().matrix_multiply(self._X_weights)).pseudo_inverse())
        if Y.num_columns() > 1:
            self._Y_rotations = self._Y_weights.matrix_multiply((self._Y_loadings.transpose().matrix_multiply(self._Y_weights)).pseudo_inverse())
        else:
            self._Y_rotations = factory.filled(1, 1, ONE)

        # Calculate regression coefficients
        y_stds: Matrix = self._standardize_Y.get_std_devs()
        self._coef = self._X_rotations.matrix_multiply(self._Y_loadings.transpose()).multiply(y_stds)

    def nipals_loop(self, X: Matrix, Y: Matrix) -> 'NipalsLoopResult':
        """
        Perform the inner NIPALS loop.

        :param X:   Predictors matrix.
        :param Y:   Response matrix.
        :return:    NipalsLoopResult.
        """
        iterations: int = 0

        y_score: Matrix = Y.get_column(0)  # (y scores)
        X_weight_old: Matrix = factory.zeros(X.num_columns(), 1)
        X_p_inv: Optional[Matrix] = None
        Y_p_inv: Optional[Matrix] = None

        eps: real = real(1e-16)

        # Repeat 1) - 3) until convergence: either change of u is lower than m_Tol or maximum
        # number of iterations has been reached (m_MaxIter)
        while True:
            # 1) Update X weights
            if self.get_weight_calculation_mode() is WeightCalculationMode.CCA:
                if X_p_inv is None:
                    # sklearn uses pinv here which ojAlgo implicitly does
                    X_p_inv = X.inverse()
                X_weight: Matrix = X_p_inv.matrix_multiply(y_score)
            else:  # PLS
                X_weight: Matrix = X.transpose().matrix_multiply(y_score).divide(y_score.norm2_squared())

            # Add eps if necessary to converge to a more acceptable solution
            if X_weight.norm2_squared() < eps:
                X_weight = X_weight.add(eps)

            # Normalize
            X_weight = X_weight.divide(X_weight.norm2() + eps)

            # 2) Calculate latent X scores
            X_score: Matrix = X.matrix_multiply(X_weight)

            # 3) Update Y weights
            if self.get_weight_calculation_mode() is WeightCalculationMode.CCA:
                if Y_p_inv is None:
                    # sklearn uses pinv here which ojAlgo implicitly does
                    Y_p_inv = Y.inverse()
                Y_weight: Matrix = Y_p_inv.matrix_multiply(X_score)
            else:  # PLS
                # WeightCalculationMode A: Regress each Y column on xscore
                Y_weight: Matrix = Y.transpose().matrix_multiply(X_score).divide(X_score.norm2_squared())

            # Normalise Y weights
            if self.norm_Y_weights:
                Y_weight = Y_weight.divide(Y_weight.norm2() + eps)

            # 4) Calculate ykScores
            Y_score: Matrix = Y.matrix_multiply(Y_weight).divide(Y_weight.norm2_squared() + eps)

            X_weight_diff: Matrix = X_weight.subtract(X_weight_old)

            if X_weight_diff.norm2_squared() < self._tol or Y.num_columns() == 1:
                break

            if iterations >= self._max_iter:
                break

            # Update stopping conditions
            X_weight_old = X_weight
            iterations += 1

        return NipalsLoopResult(X_weight, Y_weight, iterations)

    def _do_pls_predict(self, predictors: Matrix) -> Matrix:
        X: Matrix = self._standardize_X.transform(predictors)

        Y_means: Matrix = self._standardize_Y.get_means()
        Y_hat: Matrix = X.matrix_multiply(self._coef).add(Y_means)
        return Y_hat

    def _do_pls_transform(self, predictors: Matrix) -> Matrix:
        X: Matrix = self._standardize_X.transform(predictors)

        # Apply rotations
        X_scores: Matrix = X.matrix_multiply(self._X_rotations)
        return X_scores

    def do_transform_response(self, response: Matrix) -> Matrix:
        Y: Matrix = self._standardize_Y.transform(response)

        # Apply rotations
        Y_scores: Matrix = Y.matrix_multiply(self._Y_rotations)
        return Y_scores

    def get_matrix_names(self) -> List[str]:
        return ['T', 'U', 'P', 'Q']

    def get_matrix(self, name: str) -> Optional[Matrix]:
        if name == 'T':
            return self._X_scores
        elif name == 'U':
            return self._Y_scores
        elif name == 'P':
            return self._X_loadings
        elif name == 'Q':
            return self._Y_loadings
        return None

    def has_loadings(self) -> bool:
        return True

    def _do_reset(self):
        super()._do_reset()
        self._X_scores = None
        self._Y_scores = None
        self._X_loadings = None
        self._Y_loadings = None
        self._X_weights = None
        self._Y_weights = None
        self._coef = None
        self._X = None
        self._X_rotations = None
        self._Y_rotations = None
        self._standardize_X.reset()
        self._standardize_Y.reset()

    def get_loadings(self) -> Optional[Matrix]:
        return self._X_loadings

    def can_predict(self) -> bool:
        return True

    def get_weight_calculation_mode(self) -> 'WeightCalculationMode':
        return WeightCalculationMode.PLS  # Mode A in sklearn


class NipalsLoopResult:
    """
    NIPALS loop result: x and y weight matrices and number of iterations.
    """
    def __init__(self, X_weights: Matrix, Y_weights: Matrix, iterations: int):
        self.X_weights: Matrix = X_weights
        self.Y_weights: Matrix = Y_weights
        self.iterations: int = iterations


class DeflationMode(Enum):
    """
    Deflation mode enum.
    """
    CANONICAL = auto()
    REGRESSION = auto()


class WeightCalculationMode(Enum):
    """
    Mode for x/y-weight calculation
    """
    PLS = auto()
    CCA = auto()
