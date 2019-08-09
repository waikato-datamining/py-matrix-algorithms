#  _YGradientEPO.py
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

from ._YGradientGLSW import YGradientGLSW
from ...core.matrix import Matrix, factory


class YGradientEPO(YGradientGLSW):
    """
    YGradient External Parameter Orthogonalization (EPO)

    YGradientEPO is based on YGradientGLSW with the change, that the D matrix is the identity
    matrix and only a certain number of eigenvectors are kept after applying SVD.

    See also: <a href="http://wiki.eigenvector.com/index.php?title=Advanced_Preprocessing:_Multivariate_Filtering#External_Parameter_Orthogonalization_.28EPO.29">External Parameter Orthogonalization (EPO)</a>

    Parameters
    - N: Number of dominant eigenvectors to keep
    - alpha: Defines how strongly GLSW downweights interferences
    """
    def __init__(self):
        super().__init__()
        self.N: int = 5  # Number of eigenvectors to keep

    @staticmethod
    def validate_N(value: int) -> bool:
        return value > 0

    def initialize(self, x1: Optional[Matrix] = None, x2: Optional[Matrix] = None) -> str:
        if x1 is None and x2 is None:
            super().initialize()
            self.N = 5
        else:
            return super().initialize(x1, x2)

    def get_weight_matrix(self, C: Matrix) -> Matrix:
        """
        Instead of calculating D from C, create an identity matrix.

        :param C:   Covariance matrix.
        :return:    Identity matrix.
        """
        return factory.eye(self.N)

    def get_eigenvector_matrix(self, C: Matrix) -> Matrix:
        """
        Only return the first N eigenvectors.

        :param C:   Covariance matrix.
        :return:    Matrix with first N eigenvectors.
        """
        sort_dominance: bool = True
        V: Matrix = C.get_eigenvectors(sort_dominance)
        V = V.get_sub_matrix((0, V.num_rows()), (0, min(V.num_columns(), self.N)))
        return V
