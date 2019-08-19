#  _TestDataset.py
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
import os
from enum import Enum
from unittest import TestCase

from wai.ma.meta import print_stack_trace
from wai.ma.core.matrix import Matrix, helper


class TestDataset(Enum):
    """
    Test datasets that represent a file in the 'resources' folder.
    """
    BOLTS = os.path.join('resources', 'bolts.csv')
    BOLTS_RESPONSE = os.path.join('resources', 'bolts_response.csv')

    def __str__(self):
        return self.value

    def load(self) -> Matrix:
        """
        Load a matrix from a given input path.

        :return:    Matrix stored in input path.
        """
        return helper.read(self.value, True, ',')
