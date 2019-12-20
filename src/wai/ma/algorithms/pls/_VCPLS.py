#  _VCPLS.py
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
from ...core import ONE, real
from ...core.matrix import Matrix, factory
from ._PLS1 import PLS1


class VCPLS(PLS1):
    """
    Variance constrained partial least squares

    See also:
    <a href="http://or.nsfc.gov.cn/bitstream/00001903-5/485833/1/1000013952154.pdf">Variance
    constrained partial least squares</a>

    Parameters:
    - lambda: (No description given in paper)
    """
    NU: real = real(1e-7)  # The constant NU.

    def __init__(self):
        super().__init__()

        self._lambda: real = ONE

    def get_lambda(self) -> real:
        return self._lambda

    def set_lambda(self, value: real):
        self._lambda = value
        self.reset()

    lambda_ = property(get_lambda, set_lambda)

    def calculate_weights(self, x_k: Matrix, y: Matrix) -> Matrix:
        # Paper notation
        e: Matrix = x_k
        f: Matrix = y

        I: Matrix = factory.eye(e.num_columns())
        g_1: Matrix = e.t().matrix_multiply(f).matrix_multiply(f.t()).matrix_multiply(e).subtract(I.multiply(self._lambda))
        g_2: Matrix = e.t().matrix_multiply(e)

        term: Matrix = (g_2.add(I.multiply(VCPLS.NU))).inverse().matrix_multiply(g_1)

        return term.get_dominant_eigenvector()
