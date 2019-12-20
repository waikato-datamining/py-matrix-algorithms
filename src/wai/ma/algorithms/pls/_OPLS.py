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

        self._P_orth: Optional[Matrix] = None  # The P matrix
        self._T_orth: Optional[Matrix] = None  # The T matrix
        self._W_orth: Optional[Matrix] = None  # The W matrix
        self._X_osc: Optional[Matrix] = None  # Data with orthogonal signal components removed
        self._base_PLS: AbstractPLS = PLS1()  # Base PLS that is trained on the cleaned data

    def get_base_PLS(self) -> AbstractPLS:
        return self._base_PLS

    def set_base_PLS(self, value: AbstractPLS):
        self._base_PLS = value
        self.reset()

    base_PLS = property(get_base_PLS, set_base_PLS)

    def _do_reset(self):
        """
        Resets the member variables.
        """
        super()._do_reset()

        self._P_orth = None
        self._W_orth = None
        self._T_orth = None

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
                return self._P_orth
            if case('W_orth'):
                return self._W_orth
            if case('T_orth'):
                return self._T_orth
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
        return self._P_orth

    def _do_pls_configure(self, predictors: Matrix, response: Matrix):
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
        self._W_orth = factory.zeros(predictors.num_columns(), self._num_components)
        self._P_orth = factory.zeros(predictors.num_columns(), self._num_components)
        self._T_orth = factory.zeros(predictors.num_rows(), self._num_components)

        w: Matrix = X_trans.matrix_multiply(y).multiply(self.inv_L2_squared(y)).normalized()

        for current_component in range(self._num_components):
            # Calculate scores vector
            t: Matrix = X.matrix_multiply(w).multiply(self.inv_L2_squared(w))

            # Calculate loadings of X
            p: Matrix = X_trans.matrix_multiply(t).multiply(self.inv_L2_squared(t))

            # Orthogonalise weight
            w_orth: Matrix = p.subtract(w.multiply(w.transpose().matrix_multiply(p).multiply(self.inv_L2_squared(w)).as_scalar()))
            w_orth = w_orth.normalized()
            t_orth: Matrix = X.matrix_multiply(w_orth).multiply(self.inv_L2_squared(w_orth))
            p_orth: Matrix = X_trans.matrix_multiply(t_orth).multiply(self.inv_L2_squared(t_orth))

            # Remove orthogonal components from X
            X = X.subtract(t_orth.matrix_multiply(p_orth.transpose()))
            X_trans = X.transpose()

            # Store results
            self._W_orth.set_column(current_component, w_orth)
            self._T_orth.set_column(current_component, t_orth)
            self._P_orth.set_column(current_component, p_orth)

        self._X_osc = X.copy()
        self._base_PLS.configure(self._do_pls_transform(predictors), response)

    def inv_L2_squared(self, v: Matrix) -> real:
        """
        Get the inverse of the squared L2 norm.

        :param v:   Input vector.
        :return:    1.0 / norm2(v)^2.
        """
        l2: real = v.norm2_squared()
        return real(1 / l2)

    def _do_pls_transform(self, predictors: Matrix) -> Matrix:
        """
        Transforms the data.

        :param predictors:  The input data.
        :return:            The transformed data and the prediction.
        """
        # Remove signal from X_test that is orthogonal to y_train
        # X_clean = X_test - X_test*W_orth*P_orth^T
        T: Matrix = predictors.matrix_multiply(self._W_orth)
        X_orth: Matrix = T.matrix_multiply(self._P_orth.transpose())
        return predictors.subtract(X_orth)

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
        X_transformed: Matrix = self.transform(predictors)
        return self._base_PLS.predict(X_transformed)
