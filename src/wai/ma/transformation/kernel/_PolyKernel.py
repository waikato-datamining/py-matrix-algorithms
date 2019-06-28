#  _PolyKernel.py
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

from ...core import ZERO, real, NAN, ONE
from ...core.matrix import Matrix
from ._AbstractKernel import AbstractKernel


class PolyKernel(AbstractKernel):
    """
    Poly Kernel.

    K(x,y)=(gamma*x^T*y + coef_0)^d
    """
    def __init__(self):
        self.degree: int = 3
        self.coef_0: real = ZERO
        self.gamma: real = NAN
        super().__init__()

    def __setattr__(self, key, value):
        if key == 'degree' and value <= 0:
            return
        super().__setattr__(key, value)

    def apply_matrix(self, X: Matrix, Y: Optional[Matrix] = None) -> Matrix:
        if Y is None:
            Y = X
        if self.gamma is NAN:
            self.gamma = ONE / X.num_columns()
        result: Matrix = X.mul(Y.transpose())
        result = result.mul(self.gamma)
        result = result.add(self.coef_0)
        result = result.pow_elementwise(self.degree)
        return result

    def apply_vector(self, x: Matrix, y: Matrix) -> real:
        linear_term: real = x.vector_dot(y)
        if self.gamma is NAN:
            self.gamma = ONE / x.num_columns()
        return pow(self.gamma * linear_term + self.coef_0, self.degree)

    def __str__(self):
        return 'Polynomial Kernel: K(x,y)=(gamma*x^T*y + coef_0)^d, ' +\
               'gamma=' + str(self.gamma) + ', d=' + str(self.degree) + ', coef_0=' + str(self.coef_0)
