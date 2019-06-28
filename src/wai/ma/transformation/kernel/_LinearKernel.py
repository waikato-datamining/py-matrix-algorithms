#  _LinearKernel.py
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

from ...core import real
from ...core.matrix import Matrix
from ._AbstractKernel import AbstractKernel


class LinearKernel(AbstractKernel):
    """
    Linear Kernel.

    K(x_i,y_j)=x_i^T*y_j
    or also
    K(X, Y)=X*Y^T
    """
    def apply_vector(self, x: Matrix, y: Matrix) -> real:
        return x.vector_dot(y)

    def apply_matrix(self, X: Matrix, Y: Optional[Matrix] = None) -> Matrix:
        if Y is None:
            Y = X
        return X.mul(Y.transpose())

    def __str__(self):
        return 'Linear Kernel: K(x,y)=x^T*y'
