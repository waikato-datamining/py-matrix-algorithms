#  _Exponential.py
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
from typing import Optional, Tuple

from ....core.matrix.factory import filled_like
from ....core.matrix import Matrix, Axis
from ._NegEntropyApproximationFunction import NegEntropyApproximationFunction


class Exponential(NegEntropyApproximationFunction):
    """
    Exponential Negative Entropy Approximation Function.
    """
    def apply(self, x: Optional[Matrix]) -> Tuple[Matrix, Matrix]:
        x_pow_2: Matrix = x.pow(2)
        exp: Matrix = x_pow_2.divide(2).multiply(-1).exp()

        gx: Matrix = x.multiply(exp)

        ones: Matrix = filled_like(gx, 1)
        g_x: Matrix = ones.subtract(x_pow_2).multiply(exp).mean(Axis.ROWS)

        return gx, g_x
