#  _RobustScaler.py
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

from sklearn.preprocessing import RobustScaler as Intern

from ..core.algorithm import UnsupervisedMatrixAlgorithm
from ..core.matrix import Matrix


class RobustScaler(UnsupervisedMatrixAlgorithm):
    def __init__(self):
        super().__init__()

        self.with_centering: bool = True
        self.with_scaling: bool = True
        self._quantile_range: Tuple[float, float] = (25.0, 75.0)

        self._intern: Optional[Intern] = None

    def get_quantile_range(self) -> Tuple[float, float]:
        return self._quantile_range

    def set_quantile_range(self, value: Tuple[float, float]):
        if not (0.0 < value[0] < value[1] < 100.0):
            raise ValueError(f"Quantile range must be 0.0 < q_min < q_max < 100.0. "
                             f"Got q_min={value[0]}, q_max={value[1]}")

        self._quantile_range = value
        self.reset()

    quantile_range = property(get_quantile_range, set_quantile_range)

    def _do_reset(self):
        super()._do_reset()
        self._intern = None

    def _do_configure(self, data: Matrix):
        self._intern = Intern(with_centering=self.with_centering,
                              with_scaling=self.with_scaling,
                              quantile_range=self._quantile_range)
        self._intern.fit(data._data)

    def _do_transform(self, data: Matrix) -> Matrix:
        return Matrix(self._intern.transform(data._data))

    def _do_inverse_transform(self, data: Matrix) -> Matrix:
        return Matrix(self._intern.inverse_transform(data._data))
