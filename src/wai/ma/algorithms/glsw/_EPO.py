#  _EPO.py
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
from ...core.matrix import Matrix, factory
from ._GLSW import GLSW


class EPO(GLSW):
    def __init__(self):
        super().__init__()

        self._N: int = 5

    def get_N(self) -> int:
        return self._N

    def set_N(self, value: int):
        if value < 1:
            raise ValueError(f"Number of eigenvectors to keep must be at least 1 but was {value}")

        self._N = value
        self.reset()

    N = property(get_N, set_N)

    def get_weight_matrix(self, C: Matrix) -> Matrix:
        """
        Instead of calculating D from C, create an identity matrix.

        :param C:   Covariance matrix.
        :return:    Identity matrix.
        """
        return factory.eye(self._N)

    def get_eigenvector_matrix(self, C: Matrix) -> Matrix:
        """
        Only return the first N eigenvectors.

        :param C:   Covariance matrix.
        :return:    Matrix with first N eigenvectors.
        """
        sort_dominance: bool = True
        V: Matrix = C.get_eigenvectors(sort_dominance)
        V = V.get_sub_matrix((0, V.num_rows()), (0, min(V.num_columns(), self._N)))
        return V
