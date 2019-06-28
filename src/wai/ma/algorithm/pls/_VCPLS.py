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
from typing import Optional

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
        self.lambda_: real = ONE

    @staticmethod
    def validate_lambda_(value: real) -> bool:
        return True

    def initialize(self, predictors: Optional[Matrix] = None, response: Optional[Matrix] = None) -> Optional[str]:
        if predictors is None and response is None:
            super().initialize()
            self.lambda_ = ONE
        else:
            return super().initialize(predictors, response)

    def calculate_weights(self, x_k: Matrix, y: Matrix) -> Matrix:
        # Paper notation
        e: Matrix = x_k
        f: Matrix = y

        I: Matrix = factory.eye(e.num_columns())
        g_1: Matrix = e.transpose().mul(f).mul(f.transpose()).mul(e).sub(I.mul(self.lambda_))
        g_2: Matrix = e.transpose().mul(e)

        term: Matrix = (g_2.add(I.mul(VCPLS.NU))).inverse().mul(g_1)

        return term.get_dominant_eigenvector()
