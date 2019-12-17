#  _NegEntropyApproximationFunction.py
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
from typing import Optional, Tuple

from ....core.matrix import Matrix


class NegEntropyApproximationFunction(ABC):
    """
    Negative Entropy Approximation Function.
    """
    @abstractmethod
    def apply(self, x: Optional[Matrix]) -> Tuple[Matrix, Matrix]:
        """
        Apply the approximation function to the given matrix.

        :param x:   Input matrix.
        :return:    Output tuple containing approximation return value and
                    its derivative.
        """
        pass
