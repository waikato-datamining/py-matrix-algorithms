#  _SupervisedFilter.py
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
from typing import Optional

from .matrix import Matrix
from ._Filter import Filter


class SupervisedFilter(Filter):
    """
    Interface for filters which are supervised, and therefore require
    training before their transform method can be used.
    """
    @abstractmethod
    def initialize(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        """
        Initialises using the provided data.
        
        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        pass
