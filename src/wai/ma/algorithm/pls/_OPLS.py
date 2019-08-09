#  _OPLS.py
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

from wai.common import switch, default, case

from ._AbstractSingleResponsePLS import AbstractSingleResponsePLS
from ._AbstractPLS import AbstractPLS
from ._PLS1 import PLS1
from ...core import real
from ...core.matrix import Matrix, factory


class OPLS(AbstractSingleResponsePLS):
    """
    OPLS algorithm.

    See here:
    <a href="https://onlinelibrary.wiley.com/doi/pdf/10.1002/cem.695">
        Orthogonal Projections to latent structures (O-PLS)
    </a>
    """
    def __init__(self):
        super().__init__()
        self.P_orth: Optional[Matrix] = None  # The P matrix
        self.T_orth: Optional[Matrix] = None  # The T matrix
        self.W_orth: Optional[Matrix] = None  # The W matrix
        self.X_osc: Optional[Matrix] = None  # Data with orthogonal signal components removed
        self.base_PLS: Optional[AbstractPLS] = None  # Base PLS that is trained on the cleaned data

    @staticmethod
    def validate_base_PLS(value: AbstractPLS) -> bool:
        return True

    def reset(self):
        """
        Resets the member variables.
        """
        super().reset()

        self.P_orth = None
        self.W_orth = None
        self.T_orth = None

    def initialize(self, predictors: Optional[Matrix] = None, response: Optional[Matrix] = None) -> Optional[str]:
        if predictors is None and response is None:
            super().initialize()
            self.base_PLS = PLS1()
        else:
            return super().initialize(predictors, response)

    def get_matrix_names(self) -> List[str]:
        """
        Returns all the available matrices.

        :return:    The names of the matrices.
        """
        return ['P_orth',
                'W_orth',
                'T_orth']

    def get_matrix(self, name: str) -> Optional[Matrix]:
        """
        Returns the matrix with the specified name.

        :param name:    The name of the matrix.
        :return:        The matrix, None if not available.
        """
        with switch(name):
            if case('P_orth'):
                return self.P_orth
            if case('W_orth'):
                return self.W_orth
            if case('T_orth'):
                return self.T_orth
            if default():
                return None

    def has_loadings(self) -> bool:
        """
        Whether the algorithm supports return of loadings.

        :return:    True if supported.
        """
        return True

    def get_loadings(self) -> Optional[Matrix]:
        """
        Returns the loadings, if available.

        :return:    The loadings, None if not available.
        """
        return self.P_orth

    def do_perform_initialization(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        """
        Initialises using the provided data.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        X: Matrix = predictors.copy()
        X_trans: Matrix = X.transpose()
        y: Matrix = response

        # Init
        self.W_orth = factory.zeros(predictors.num_columns(), self.num_components)
        self.P_orth = factory.zeros(predictors.num_columns(), self.num_components)
        self.T_orth = factory.zeros(predictors.num_rows(), self.num_components)

        w: Matrix = X_trans.mul(y).mul(self.inv_L2_squared(y)).normalized()

        for current_component in range(self.num_components):
            # Calculate scores vector
            t: Matrix = X.mul(w).mul(self.inv_L2_squared(w))

            # Calculate loadings of X
            p: Matrix = X_trans.mul(t).mul(self.inv_L2_squared(t))

            # Orthogonalise weight
            w_orth: Matrix = p.sub(w.mul(w.transpose().mul(p).mul(self.inv_L2_squared(w)).as_real()))
            w_orth = w_orth.normalized()
            t_orth: Matrix = X.mul(w_orth).mul(self.inv_L2_squared(w_orth))
            p_orth: Matrix = X_trans.mul(t_orth).mul(self.inv_L2_squared(t_orth))

            # Remove orthogonal components from X
            X = X.sub(t_orth.mul(p_orth.transpose()))
            X_trans = X.transpose()

            # Store results
            self.W_orth.set_column(current_component, w_orth)
            self.T_orth.set_column(current_component, t_orth)
            self.P_orth.set_column(current_component, p_orth)

        self.X_osc = X.copy()
        self.base_PLS.initialize(self.do_transform(predictors), response)

        return None

    def inv_L2_squared(self, v: Matrix) -> real:
        """
        Get the inverse of the squared L2 norm.

        :param v:   Input vector.
        :return:    1.0 / norm2(v)^2.
        """
        l2: real = v.norm2_squared()
        return real(1 / l2)

    def do_transform(self, predictors: Matrix) -> Matrix:
        """
        Transforms the data.

        :param predictors:  The input data.
        :return:            The transformed data and the prediction.
        """
        # Remove signal from X_test that is orthogonal to y_train
        # X_clean = X_test - X_test*W_orth*P_orth^T
        T: Matrix = predictors.mul(self.W_orth)
        X_orth: Matrix = T.mul(self.P_orth.transpose())
        return predictors.sub(X_orth)

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
        X_transformed: Matrix = self.transform(predictors)
        return self.base_PLS.predict(X_transformed)
