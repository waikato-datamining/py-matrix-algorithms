#  _SupervisedMatrixAlgorithm.py
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
from abc import abstractmethod

from ..matrix import Matrix
from ._ConfiguredMatrixAlgorithm import ConfiguredMatrixAlgorithm


class SupervisedMatrixAlgorithm(ConfiguredMatrixAlgorithm):
    """
    Base class for algorithms that are configured on a feature
    and target matrix.
    """
    def configure_and_transform(self, X: Matrix, y: Matrix) -> Matrix:
        """
        Transforms the feature matrix, configuring the algorithm on the matrices
        if it is not already configured.

        :param X:   The feature matrix to configure on and apply the algorithm to.
        :param y:   The target matrix to configure on.
        :return:    The matrix resulting from the transformation.
        """
        # Configure on first pair of matrices seen if not done explicitly
        if not self.is_configured():
            self.configure(X, y)

        return self.transform(X)

    def configure(self, X: Matrix, y: Matrix):
        """
        Configures this algorithm on the given feature and target matrices.

        :param X:   The feature configuration matrix.
        :param y:   The target configuration matrix.
        """
        # Check that a configuration matrix was given
        if X is None:
            raise ValueError("Cannot configure on null feature matrix")
        if y is None:
            raise ValueError("Cannot configure on null target matrix")

        # Perform actual configuration
        self._do_configure(X, y)

        # Flag that we are configured
        self._set_configured()

    @abstractmethod
    def _do_configure(self, X: Matrix, y: Matrix):
        """
        Configuration-specific implementation. Override to configure
        the algorithm on the given matrices.

        :param X:   The feature configuration matrix.
        :param y:   The target configuration matrix.
        """
        pass
