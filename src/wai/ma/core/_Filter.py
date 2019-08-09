#  _Filter.py
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

from .matrix import Matrix


class Filter(ABC):
    """
    Filter API that exposes a method transform(Matrix) which takes
    a matrix and returns a matrix based on this filter's transformation rules.
    """
    @abstractmethod
    def transform(self, predictors: Matrix) -> Matrix:
        """
        Transform a given matrix into another matrix based on the
        filter's implementation.

        :param predictors:  Input matrix.
        :return:            Transformed matrix.
        """
        pass
