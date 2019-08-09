#  _RBFKernel.py
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
from ...core import ONE, real, NAN
from ...core.matrix import Matrix
from ...core.utils import exp
from ._AbstractKernel import AbstractKernel


class RBFKernel(AbstractKernel):
    """
    Radial Basis Function Kernel.

    K(x,y) = exp(-1*||x - y||^2/(2*sigma^2))
    or
    K(x,y) = exp(-1*gamma*||x - y||^2), with gamma=2*sigma^2
    """
    def __init__(self):
        self.gamma: real = ONE  # Gamma parameter
        super().__init__()

    def apply_vector(self, x: Matrix, y: Matrix) -> real:
        norm_2: real = x.sub(y).norm2()
        if self.gamma is NAN:
            self.gamma = ONE / x.num_columns()
        return exp(-ONE * self.gamma * norm_2)

    def __str__(self):
        return 'RBF Kernel: K(x,y) = exp(-1*gamma*||x - y||^2), gamma=' + str(self.gamma)
