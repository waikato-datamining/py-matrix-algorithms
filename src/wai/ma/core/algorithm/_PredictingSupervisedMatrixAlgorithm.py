#  _PredictingSupervisedMatrixAlgorithm.py
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
from abc import ABC, abstractmethod

from ..matrix import Matrix
from ._SupervisedMatrixAlgorithm import SupervisedMatrixAlgorithm


class PredictingSupervisedMatrixAlgorithm(SupervisedMatrixAlgorithm, ABC):
    """
    Base class for algorithms that, once configured, can predict
    target values from feature matrices.
    """
    def configure_and_predict(self, X: Matrix, y: Matrix) -> Matrix:
        """
        Performs predictions on the feature matrix, configuring the
        algorithm on the matrices if it is not already configured.

        :param X:   The feature matrix to configure on and predict against.
        :param y:   The target matrix to configure on.
        :return:    The predictions.
        """
        # Configure on first pair of matrices seen if not done explicitly
        if not self.is_configured():
            self.configure(X, y)

        return self.predict(X)

    def predict(self, X: Matrix) -> Matrix:
        """
        Performs predictions on the feature matrix.

        :param X:   The feature matrix to predict against.
        :return:    The predictions.
        """
        if X is None:
            raise ValueError("Can't predict against null feature matrix")

        # Ensure the algorithm is configured
        self.ensure_configured()

        return self._do_predict(X)

    @abstractmethod
    def _do_predict(self, X: Matrix) -> Matrix:
        """
        Prediction-specific implementation. Override to predict target
        values for the given feature matrix.

        :param X:   The feature matrix to predict against.
        :return:    The predictions.
        """
        pass
