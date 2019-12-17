#  _PowerTransformer.py
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

from sklearn.preprocessing import PowerTransformer as Intern

from ..core.algorithm import UnsupervisedMatrixAlgorithm
from ..core.matrix import Matrix


class PowerTransformer(UnsupervisedMatrixAlgorithm):
    # The methods that can be used
    YEO_JOHNSON = "yeo-johnson"
    BOX_COX = "box-cox"

    def __init__(self):
        super().__init__()

        self._method: str = PowerTransformer.YEO_JOHNSON
        self.standardise: bool = True

        self._intern: Optional[Intern] = None

    def get_method(self) -> str:
        return self._method

    def set_method(self, value: str):
        available_methods = {PowerTransformer.YEO_JOHNSON, PowerTransformer.BOX_COX}
        if value not in available_methods:
            raise ValueError(f"Method was {value} but must be one of: " +
                             ", ".join(PowerTransformer.METHODS))

        self._method = value
        self.reset()

    method = property(get_method, set_method)

    def _do_reset(self):
        super()._do_reset()
        self._intern = None

    def _do_configure(self, data: Matrix):
        self._intern = Intern(method=self._method, standardize=self.standardise)
        self._intern.fit(data._data)

    def _do_transform(self, data: Matrix) -> Matrix:
        return Matrix(self._intern.transform(data._data))

    def _do_inverse_transform(self, data: Matrix) -> Matrix:
        return Matrix(self._intern.inverse_transform(data._data))
