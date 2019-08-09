#  _MultiFilter.py
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
from typing import List

from .matrix import Matrix
from ._Filter import Filter


class MultiFilter(Filter):
    """
    Filter which encapsulates a series of sub-filters, and applies
    them in a given order.
    """
    def __init__(self, filters: List[Filter]):
        # The filters in the order of application
        self.filters: List[Filter] = filters
        
    def transform(self, predictors: Matrix) -> Matrix:
        # The result starts as the predictors
        result: Matrix = predictors
        
        # Apply each filter in ordered turn
        for filter in self.filters:
            result = filter.transform(result)
            
        return result
