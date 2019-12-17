#  _UnsupervisedMatrixAlgorithm.py
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
from ._ConfiguredMatrixAlgorithm import ConfiguredMatrixAlgorithm


class UnsupervisedMatrixAlgorithm(ConfiguredMatrixAlgorithm, ABC):
    """
    Base class for algorithms that are configured on a feature
    matrix.
    """
    def configure_and_transform(self, X: Matrix) -> Matrix:
        """
        Transforms the data, configuring the algorithm on the matrix
        if it is not already configured.

        :param X:   The matrix to configure on and apply the algorithm to.
        :return:    The matrix resulting from the transformation.
        """
        # Configure on first matrix seen if not done explicitly
        if not self.is_configured():
            self.configure(X)

        return self.transform(X)

    def configure(self, X: Matrix):
        """
        Configures this algorithm on the given matrix.

        :param X:   The configuration matrix.
        """
        # Check that a configuration matrix was given
        if X is None:
            raise ValueError("Cannot configure on null matrix")

        # Perform actual configuration
        self._do_configure(X)

        # Flag that we are configured
        self._set_configured()

    @abstractmethod
    def _do_configure(self, X: Matrix):
        """
        Configuration-specific implementation. Override to configure
        the algorithm on the given matrix.

        :param X:   The configuration matrix.
        """
        pass
