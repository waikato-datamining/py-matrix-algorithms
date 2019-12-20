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
    def get_weight_calculation_mode(self) -> WeightCalculationMode:
        return WeightCalculationMode.CCA  # Mode B in sklearn

    def get_deflation_mode(self) -> DeflationMode:
        return DeflationMode.CANONICAL

    def set_deflation_mode(self, value: DeflationMode):
        if value is not DeflationMode.CANONICAL:
            raise ValueError("CCARegression only allows CANONICAL deflation mode")

        super().set_deflation_mode(value)

    deflation_mode = property(get_deflation_mode, set_deflation_mode)
