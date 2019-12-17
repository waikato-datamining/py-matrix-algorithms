#  _RowNorm.py
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

from ..core import real
from ..core.algorithm import MatrixAlgorithm
from ..core.matrix import Matrix, Axis


class RowNorm(MatrixAlgorithm):
    def _do_transform(self, data: Matrix) -> Matrix:
        # Get the row means and standard deviations
        means: Matrix = data.mean(Axis.ROWS)
        std_devs: Matrix = data.standard_deviation(Axis.ROWS)

        # Avoid divide-by-zero error
        std_devs.apply_elementwise(lambda v: v if v != 0 else real(1), in_place=True)

        # Normalise
        return data.subtract(means).divide(std_devs, in_place=True)

    def is_non_invertible(self) -> bool:
        return True
