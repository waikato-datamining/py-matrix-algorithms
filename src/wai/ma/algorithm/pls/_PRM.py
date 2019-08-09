#  _PRM.py
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
from typing import Optional, List

from wai.common import switch, case

from ._AbstractSingleResponsePLS import AbstractSingleResponsePLS
from ._SIMPLS import SIMPLS
from ...core import real, ZERO, ONE
from ...core.matrix import Matrix, factory
from ...core.utils import sqrt


class PRM(AbstractSingleResponsePLS):
    """
    Partial robust M-regression as described in
    <a href="https://www.sciencedirect.com/science/article/abs/pii/S0169743905000638">
        Partial robust M-regression
    </a>

    Parameters:
    - c: Tuning constant for Fair weight function. Higher values result in a
         flatter function. c=Infinity is equal to SIMPLS result.
    - tol: Iterative convergence tolerance
    - maxIter: Maximum number of iterations
    - numSimplsCoefficients: Number of SIMPLS coefficients
    """
    def __init__(self):
        super().__init__()
        self.tol: real = ZERO  # Loop improvement tolerance
        self.max_iter: int = 0  # Loop maximum number of iterations
        self.C: real = ONE  # C parameter
        self.W_r: Optional[Matrix] = None  # Residual weights N x 1
        self.W_x: Optional[Matrix] = None  # Leverage weights N x 1
        self.T: Optional[Matrix] = None  # T score matrix from SIMPLS algorithm
        self.gamma: Optional[Matrix] = None  # Gamma regression coefficients from SIMPLS algorithm
        self.num_SIMPLS_coefficients: int = 0  # The number of SIMPLS coefficients in W to keep (0 keep all)
        self.final_regression_coefficients: Optional[Matrix] = None  # Final regression coefficients
        self.simpls: Optional[SIMPLS] = None  # SIMPLS algorithm

    @staticmethod
    def validate_num_SIMPLS_coefficients(value: int) -> bool:
        return True

    @staticmethod
    def validate_max_iter(value: int) -> bool:
        return value >= 0

    @staticmethod
    def validate_tol(value: real) -> bool:
        return value >= 0

    @staticmethod
    def validate_C(value: real) -> bool:
        return abs(value) >= 1e-10

    def reset(self):
        super().reset()
        self.W_r = None
        self.W_x = None
        self.final_regression_coefficients = None
        self.gamma = None
        self.T = None
        self.simpls = None

    def initialize(self, predictors: Optional[Matrix] = None, response: Optional[Matrix] = None) -> Optional[str]:
        if predictors is None and response is None:
            super().initialize()
            self.C = real(4)
            self.tol = real(1e-6)
            self.max_iter = 500
            self.num_SIMPLS_coefficients = -1
        else:
            return super().initialize(predictors, response)

    def get_matrix_names(self) -> List[str]:
        return ['B',
                'Wr',
                'Wx',
                'W']

    def get_matrix(self, name: str) -> Optional[Matrix]:
        with switch(name):
            if case('B'):
                return self.final_regression_coefficients
            if case('Wr'):
                return self.W_r
            if case('Wx'):
                return self.W_x
            if case('W'):
                return self.W_r.mul_elementwise(self.W_x)
        return None

    def has_loadings(self) -> bool:
        return False

    def get_loadings(self) -> Optional[Matrix]:
        return None

    def to_string(self) -> str:
        return ''

    def fair_function(self, z: real, c: real) -> real:
        """
        Fair function implementation.

        f(z,c) = 1/(1+|z/c|)^2

        :param z:   First parameter.
        :param c:   Second parameter.
        :return:    Fair function result.
        """
        return real(1.0 / pow(1.0 + abs(z / c), 2))

    def init_weights(self, X: Matrix, y: Matrix):
        """
        Initialise the residual and leverage weights.

        :param X:   Predictor matrix.
        :param y:   Response matrix.
        """
        self.update_residual_weights(X, y)
        self.update_leverage_weights(X)

    def update_leverage_weights(self, T: Matrix):
        """
        Update the leverage weights based on the score matrix T.

        :param T:   Score matrix.
        """
        n: int = T.num_rows()
        self.W_x = factory.zeros(n, 1)

        row_L1_median: Matrix = self.geometric_median(T)
        distances_to_median: Matrix = self.c_dist(T, row_L1_median)

        median_of_dists_to_median: real = distances_to_median.median()

        # Calculate w_xi by f(zi, c) with zi = (distance_i to median) / (median of distances to median)
        for i in range(n):
            dist_to_median: real = distances_to_median.get(i, 0)
            w_xi: real = self.fair_function(real(dist_to_median / median_of_dists_to_median), self.C)
            self.W_x.set(i, 0, w_xi)

    def update_residual_weights(self, X: Matrix, y: Matrix):
        """
        Update the residual weights based on the new residuals.

        :param X:   Predictors.
        :param y:   Response.
        """
        n: int = X.num_rows()
        self.W_r = factory.zeros(n, 1)

        residuals: Matrix = factory.zeros(n, 1)

        # Check if this is the first iteration
        is_first_iteration: bool = self.T is None and self.gamma is None

        y_i_hat: real = y.median()
        for i in range(n):
            # Use t_i * gamma as estimation if iteration > 0
            if not is_first_iteration:
                y_i_hat = self.T.get_row(i).mul(self.gamma).as_real()

            # Calculate residual
            y_i: real = y.get(i, 0)
            r_i: real = real(y_i - y_i_hat)
            residuals.set(i, 0, r_i)

        # Get estimate of residual scale
        sigma: real = self.median_absolute_deviation(residuals)
        residuals = residuals.div(sigma)

        # Calculate weights
        for i in range(n):
            r_i: real = residuals.get(i, 0)
            w_ri: real = self.fair_function(r_i, self.C)
            self.W_r.set(i, 0, w_ri)

    def median_absolute_deviation(self, v: Matrix) -> real:
        """
        Mean Absolute Deviation:
        MAD(v) = median_i | v_i - median_j v_j |

        :param v:   Input vector.
        :return:    MAD result.
        """
        return v.sub(v.median()).abs().median()

    def do_perform_initialization(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        """
        Trains using the provided data.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        X: Matrix = predictors
        y: Matrix = response
        U: Optional[Matrix] = None

        # If X: n x p and p > n, use SVD to replace X with n x n matrix
        # See also: Remark 2 in paper
        has_more_columns_than_rows: bool = X.num_columns() > X.num_rows()
        if has_more_columns_than_rows:
            X_t: Matrix = X.transpose()
            U = X_t.svd_U()
            V: Matrix = X_t.svd_V()
            S: Matrix = X_t.svd_S()

            # Replace X with n x n matrix
            X = V.mul(S)

        # 1) Compute robust starting values fro residual and leverage weights
        self.init_weights(X, y)

        gamma_old: Optional[Matrix] = None
        num_components: int = self.num_components
        self.gamma = factory.zeros(num_components, 1)
        iteration: int = 0

        # Loop until convergence of gamma
        while iteration == 0 or (self.gamma.sub(gamma_old).norm2_squared() < self.tol and iteration < self.max_iter):
            # 2) Perform PLS (SIMPLS) on reweighted data matrices
            X_p: Matrix = self.get_reweighted_matrix(X)
            y_p: Matrix = self.get_reweighted_matrix(y)

            self.simpls = SIMPLS()
            self.simpls.num_coefficients = self.num_SIMPLS_coefficients
            self.simpls.num_components = num_components
            self.simpls.initialize(X_p, y_p)

            # Get scores and regression coefficients
            gamma_old = self.gamma.copy()
            self.T = self.simpls.transform(X_p)
            self.gamma = self.simpls.get_matrix('Q').transpose()

            # Rescale t_i by 1/sqrt(w_i)
            for i in range(self.T.num_rows()):
                w_i_sqrt: real = sqrt(self.get_combined_weight(i))
                row_i_scaled: Matrix = self.T.get_row(i).div(w_i_sqrt)
                self.T.set_row(i, row_i_scaled)

            # Update weights
            self.update_residual_weights(X_p, y_p)
            self.update_leverage_weights(self.T)
            iteration += 1

        # Get the final regression coefficients from the latest SIMPLS run
        self.final_regression_coefficients = self.simpls.get_matrix('B')

        # If X has been replaced by US, the regression coefficients need to be
        # back-transformed into beta_hat = U*beta_p
        if has_more_columns_than_rows:
            self.final_regression_coefficients = U.mul(self.final_regression_coefficients)

        return None

    def can_predict(self) -> bool:
        return True

    def do_transform(self, predictors: Matrix) -> Matrix:
        return predictors.mul(self.simpls.get_matrix('W'))

    def do_perform_predictions(self, predictors: Matrix) -> Matrix:
        return predictors.mul(self.final_regression_coefficients)

    def get_reweighted_matrix(self, A: Matrix) -> Matrix:
        return A.copy().scale_by_column_vector(self.W_r.mul_elementwise(self.W_x).sqrt())

    def get_combined_weight(self, i: int) -> real:
        w_xi: real = self.W_x.get_row(i).as_real()
        w_ri: real = self.W_r.get_row(i).as_real()
        return real(w_xi * w_ri)

    def geometric_median(self, X: Matrix) -> Matrix:
        """
        Geometric median according to
        <a href="https://en.wikipedia.org/wiki/Geometric_median">
            Geometric Median
        </a>

        Weiszfeld's algorithm.

        :param X:   Points.
        :return:    Geometric median of X.
        """
        # Initial guess
        guess: Matrix = X.mean(0)

        iteration: int = 0
        while iteration < self.max_iter:
            dists: Matrix = self.c_dist(X, guess)

            def _(value):
                if abs(value) < 1e-10:
                    return real(1.0 / 0.1)  # Fix zero distances
                else:
                    return real(1.0 / value)  # Invert
            dists = dists.apply_elementwise(_)

            nom: Matrix = X.scale_by_column_vector(dists).sum(0)
            denom: real = dists.sum(0).as_real()
            guess_next: Matrix = nom.div(denom)

            change: real = guess_next.sub(guess).norm2_squared()
            guess = guess_next
            if change < self.tol:
                break

            iteration += 1

        return guess

    def c_dist(self, X: Matrix, vector: Matrix) -> Matrix:
        """
        Distance function between all rows of X and a given row vector.

        :param X:       Input matrix with rows.
        :param vector:  Row vector to compare all rows of X to.
        :return:        Distances of each row r_i with the input vector.
        """
        dist: Matrix = factory.zeros(X.num_rows(), 1)
        for i in range(X.num_rows()):
            row_i: Matrix = X.get_row(i)
            d: real = row_i.sub(vector).norm2()
            dist.set(i, 0, d)
        return dist
