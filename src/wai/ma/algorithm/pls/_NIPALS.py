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
from ...core import ZERO, real, ONE
from ...core.matrix import Matrix, factory
from ...transformation import Standardize


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
        self.X_scores: Optional[Matrix] = None  # Scores on X
        self.Y_scores: Optional[Matrix] = None  # Scores on Y
        self.X_loadings: Optional[Matrix] = None  # Loadings on X
        self.Y_loadings: Optional[Matrix] = None  # Loadings on Y
        self.X_weights: Optional[Matrix] = None  # Weights on X
        self.Y_weights: Optional[Matrix] = None  # Weights on Y
        self.X_rotations: Optional[Matrix] = None  # Projection of X into latent space
        self.Y_rotations: Optional[Matrix] = None  # Projection of Y into latent space
        self.X: Optional[Matrix] = None  # Training points
        self.coef: Optional[Matrix] = None  # Regression coefficients
        self.tol: real = real(1e-6)  # Inner NIPALS loop improvement tolerance
        self.max_iter: int = 500  # Inner NIPALS loop maximum number of iterations
        self.norm_Y_weights: bool = False  # Flag to normalize Y weights
        self.standardize_X: Optional[Standardize] = None  # Standarize X tranformation
        self.standardize_Y: Optional[Standardize] = None  # Standardize Y transformation
        self.deflation_mode: DeflationMode = DeflationMode.REGRESSION

    def initialize(self, predictors: Optional[Matrix] = None, response: Optional[Matrix] = None) -> Optional[str]:
        if predictors is None and response is None:
            super().initialize()
            self.tol = real(1e-6)
            self.max_iter = 500
            self.norm_Y_weights = False
            self.standardize_X = Standardize()
            self.standardize_Y = Standardize()
            self.deflation_mode = DeflationMode.REGRESSION
        else:
            return super().initialize(predictors, response)

    @staticmethod
    def validate_max_iter(value: int) -> bool:
        return value >= 0

    @staticmethod
    def validate_tol(value: real) -> bool:
        return value >= ZERO

    def get_min_columns_response(self) -> int:
        return 1

    def get_max_columns_response(self) -> int:
        return -1

    def do_perform_initialization(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        # Init
        X: Matrix = predictors
        X = self.standardize_X.transform(X)
        Y: Matrix = response
        Y = self.standardize_Y.transform(Y)

        # Dimensions
        num_rows: int = X.num_rows()
        num_features: int = X.num_columns()
        num_classes: int = Y.num_columns()
        num_components: int = self.num_components

        # Init matrices
        self.X_scores = factory.zeros(num_rows, num_components)  # T
        self.Y_scores = factory.zeros(num_rows, num_components)  # U

        self.X_weights = factory.zeros(num_features, num_components)  # W
        self.Y_weights = factory.zeros(num_classes, num_components)  # C

        self.X_loadings = factory.zeros(num_features, num_components)  # P
        self.Y_loadings = factory.zeros(num_classes, num_components)  # Q

        yk_loading: Matrix = factory.zeros(num_classes, 1)

        eps: real = real(1e-10)
        for k in range(num_components):
            if Y.transpose().mul(Y).all(lambda e: e < eps):
                self.logger.warning('Y residual constant at iteration ' + str(k))
                break

            res: NipalsLoopResult = self.nipals_loop(X, Y)
            xk_weight: Matrix = res.X_weights
            yk_weight: Matrix = res.Y_weights

            # Calculate latent X and Y scores
            xk_score: Matrix = X.mul(xk_weight)
            yk_score: Matrix = Y.mul(yk_weight).div(yk_weight.norm2_squared())

            if xk_score.norm2_squared() < eps:
                self.logger.warning('X scores are null at component ' + str(k))
                break

            # Deflate X
            xk_loading: Matrix = X.transpose().mul(xk_score).div(xk_score.norm2_squared())
            X = X.sub(xk_score.mul(xk_loading.transpose()))

            # Deflate Y
            if self.deflation_mode is DeflationMode.CANONICAL:
                yk_loading: Matrix = Y.transpose().mul(yk_score).div(yk_score.norm2_squared())
                Y = Y.sub(yk_score.mul(yk_loading.transpose()))
            elif self.deflation_mode is DeflationMode.REGRESSION:
                yk_loading: Matrix = Y.transpose().mul(xk_score).div(xk_score.norm2_squared())
                Y = Y.sub(xk_score.mul(yk_loading.transpose()))

            # Store results
            self.X_scores.set_column(k, xk_score)
            self.Y_scores.set_column(k, yk_score)
            self.X_weights.set_column(k, xk_weight)
            self.Y_weights.set_column(k, yk_weight)
            self.X_loadings.set_column(k, xk_loading)
            self.Y_loadings.set_column(k, yk_loading)

        self.X = X
        self.X_rotations = self.X_weights.mul((self.X_loadings.transpose().mul(self.X_weights)).pseudo_inverse())
        if Y.num_columns() > 1:
            self.Y_rotations = self.Y_weights.mul((self.Y_loadings.transpose().mul(self.Y_weights)).pseudo_inverse())
        else:
            self.Y_rotations = factory.filled(1, 1, ONE)

        # Calculate regression coefficients
        y_stds: Matrix = self.standardize_Y.get_std_devs()
        self.coef = self.X_rotations.mul(self.Y_loadings.transpose()).scale_by_row_vector(y_stds)
        return None

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
                X_weight: Matrix = X_p_inv.mul(y_score)
            else:  # PLS
                X_weight: Matrix = X.transpose().mul(y_score).div(y_score.norm2_squared())

            # Add eps if necessary to converge to a more acceptable solution
            if X_weight.norm2_squared() < eps:
                X_weight = X_weight.add(eps)

            # Normalize
            X_weight = X_weight.div(X_weight.norm2() + eps)

            # 2) Calculate latent X scores
            X_score: Matrix = X.mul(X_weight)

            # 3) Update Y weights
            if self.get_weight_calculation_mode() is WeightCalculationMode.CCA:
                if Y_p_inv is None:
                    # sklearn uses pinv here which ojAlgo implicitly does
                    Y_p_inv = Y.inverse()
                Y_weight: Matrix = Y_p_inv.mul(X_score)
            else:  # PLS
                # WeightCalculationMode A: Regress each Y column on xscore
                Y_weight: Matrix = Y.transpose().mul(X_score).div(X_score.norm2_squared())

            # Normalise Y weights
            if self.norm_Y_weights:
                Y_weight = Y_weight.div(Y_weight.norm2() + eps)

            # 4) Calculate ykScores
            Y_score: Matrix = Y.mul(Y_weight).div(Y_weight.norm2_squared() + eps)

            X_weight_diff: Matrix = X_weight.sub(X_weight_old)

            if X_weight_diff.norm2_squared() < self.tol or Y.num_columns() == 1:
                break

            if iterations >= self.max_iter:
                break

            # Update stopping conditions
            X_weight_old = X_weight
            iterations += 1

        return NipalsLoopResult(X_weight, Y_weight, iterations)

    def do_perform_predictions(self, predictors: Matrix) -> Matrix:
        X: Matrix = self.standardize_X.transform(predictors)

        Y_means: Matrix = self.standardize_Y.get_means()
        Y_hat: Matrix = X.mul(self.coef).add_by_vector(Y_means)
        return Y_hat

    def do_transform(self, predictors: Matrix) -> Matrix:
        X: Matrix = self.standardize_X.transform(predictors)

        # Apply rotations
        X_scores: Matrix = X.mul(self.X_rotations)
        return X_scores

    def do_transform_response(self, response: Matrix) -> Matrix:
        Y: Matrix = self.standardize_Y.transform(response)

        # Apply rotations
        Y_scores: Matrix = Y.mul(self.Y_rotations)
        return Y_scores

    def get_matrix_names(self) -> List[str]:
        return ['T', 'U', 'P', 'Q']

    def get_matrix(self, name: str) -> Optional[Matrix]:
        if name == 'T':
            return self.X_scores
        elif name == 'U':
            return self.Y_scores
        elif name == 'P':
            return self.X_loadings
        elif name == 'Q':
            return self.Y_loadings
        return None

    def has_loadings(self) -> bool:
        return True

    def reset(self):
        super().reset()
        self.X_scores = None
        self.Y_scores = None
        self.X_loadings = None
        self.Y_loadings = None
        self.X_weights = None
        self.Y_weights = None
        self.coef = None
        self.X = None
        self.X_rotations = None
        self.Y_rotations = None
        self.standardize_X = Standardize()
        self.standardize_Y = Standardize()

    def get_loadings(self) -> Optional[Matrix]:
        return self.X_loadings

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
