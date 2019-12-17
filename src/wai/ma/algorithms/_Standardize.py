#  _Standardize.py
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

from typing import Optional, IO

from ..core import real, Serialisable
from ..core.algorithm import UnsupervisedMatrixAlgorithm
from ..core.matrix import Matrix, Axis


class Standardize(UnsupervisedMatrixAlgorithm, Serialisable):
    def __init__(self):
        super().__init__()

        self._means: Optional[Matrix] = None
        self._std_devs: Optional[Matrix] = None

    def _do_reset(self):
        super().reset()

        self._means = None
        self._std_devs = None

    def _do_configure(self, data: Matrix):
        self._means = data.mean(Axis.COLUMNS)
        self._std_devs = data.standard_deviation(Axis.COLUMNS)

        # Make sure we don't do a divide by zero
        self._std_devs.apply_elementwise(lambda v: v if v != 0 else real(1), in_place=True)

    def _do_transform(self, data: Matrix) -> Matrix:
        return data.subtract(self._means).divide(self._std_devs, in_place=True)

    def _do_inverse_transform(self, data: Matrix) -> Matrix:
        return data.multiply(self._std_devs).add(self._means, in_place=True)

    def get_means(self) -> Matrix:
        self.ensure_configured()

        return self._means.copy()

    def get_std_devs(self) -> Matrix:
        self.ensure_configured()

        return self._std_devs.copy()

    def serialise_state(self, stream: IO[bytes]):
        # Can't serialise our state until we've been configured
        self.ensure_configured()

        # Serialise out our mean and standard deviation coefficient matrices
        self._means.serialise_state(stream)
        self._std_devs.serialise_state(stream)
