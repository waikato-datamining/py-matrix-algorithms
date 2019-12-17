#  _Center.py
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

from ..core.algorithm import UnsupervisedMatrixAlgorithm
from ..core.matrix import Matrix, Axis


class Center(UnsupervisedMatrixAlgorithm):
    """
    Centers the data in the matrix columns according to the mean.
    """
    def __init__(self):
        super().__init__()

        self._means: Optional[Matrix] = None

    def _do_reset(self):
        self._means = None

    def _do_configure(self, X: Matrix):
        self._means = X.mean(Axis.COLUMNS)

        if self.debug:
            self.logger.info("Means: " + self._means.row_str(0))

    def _do_transform(self, X: Matrix) -> Matrix:
        return X.subtract(self._means)

    def _do_inverse_transform(self, X: Matrix) -> Matrix:
        return X.add(self._means)
