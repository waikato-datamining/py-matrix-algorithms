#  _ConfiguredMatrixAlgorithm.py
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

from ..errors import UnconfiguredAlgorithmError
from ..matrix import Matrix
from ._MatrixAlgorithm import MatrixAlgorithm


class ConfiguredMatrixAlgorithm(MatrixAlgorithm, ABC):
    """
    Base class for algorithms that require configuration matrices
    before they can perform their transformation. Algorithms themselves
    should not sub-class this class, but instead sub-classes of this
    class which expose the configuration method.
    """
    def __init__(self):
        super().__init__()

        self.__configured: bool = False

    def reset(self):
        """
        Resets the algorithm to its unconfigured state.
        """
        self._do_reset()
        self.__configured = False

    @abstractmethod
    def _do_reset(self):
        """
        Resets the algorithm to its unconfigured state. Override
        to reset configured algorithm state.
        """
        pass

    def _set_configured(self):
        """
        Allows sub-classes to set the configured flag
        once they have been configured. Package-private
        so that algorithms themselves don't set this, but
        the specific configuration sub-types do on their
        behalf.
        """
        self.__configured = True

    def is_configured(self) -> bool:
        """
        Whether this algorithm has been configured.

        :return:    True if the algorithm is configured,
                    false if not.
        """
        return self.__configured

    def ensure_configured(self):
        """
        Raises UnconfiguredAlgorithmError if this
        algorithm hasn't been configured yet.

        :raises UnconfiguredAlgorithmError:     If the algorithm
                                                is not configured.
        """
        if not self.is_configured():
            raise UnconfiguredAlgorithmError(self.__class__)

    def transform(self, X: Matrix) -> Matrix:
        # Ensure the algorithm is configured
        self.ensure_configured()

        return super().transform(X)

    def inverse_transform(self, X: Matrix) -> Matrix:
        # Ensure the algorithm is configured
        self.ensure_configured()

        return super().inverse_transform(X)

    def is_non_invertible(self) -> bool:
        # Non-configured algorithms are never invertible
        return not self.is_configured()
