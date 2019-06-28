#  _CCARegression.py
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
from ._NIPALS import NIPALS, WeightCalculationMode, DeflationMode


class CCARegression(NIPALS):
    def get_weight_calculation_mode(self) -> 'WeightCalculationMode':
        return WeightCalculationMode.CCA  # Mode B in sklearn

    def __getattribute__(self, item):
        if item == 'deflation_mode':
            return DeflationMode.CANONICAL
        else:
            return super().__getattribute__(item)

    def __setattr__(self, key, value):
        if key == 'deflation_mode':
            if value is not DeflationMode.CANONICAL:
                self.logger.warning('CCARegression only allows CANONICAL deflation mode.')
        else:
            super().__setattr__(key, value)
