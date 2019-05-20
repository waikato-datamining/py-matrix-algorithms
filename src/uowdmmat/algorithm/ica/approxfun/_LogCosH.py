#  _LogCosH.py
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
from typing import Tuple, Optional

from ....core import real
from ....core.matrix import Matrix
from ....core.matrix.factory import zeros, filled
from ._NegEntropyApproximationFunction import NegEntropyApproximationFunction


class LogCosH(NegEntropyApproximationFunction):
    """
    LogCosH Negative Entropy Approximation Function.
    """
    def __init__(self, alpha: real = real(1.0)):
        super().__init__()
        self.alpha = alpha

    def apply(self, x: Optional[Matrix]) -> Tuple[Matrix, Matrix]:
        x = x.mul(self.alpha)
        gx: Matrix = x.tanh()
        g_x: Matrix = zeros(gx.num_rows(), 1)
        ones: Matrix = filled(1, gx.num_columns(), 1.0)
        for i in range(gx.num_rows()):
            gxi: Matrix = gx.get_row(i)
            g_xi = ones.sub(gxi.pow_elementwise(2)).mul(self.alpha).mean()
            g_x.set(i, 0, g_xi)
        return gx, g_x
