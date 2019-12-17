#  _Log.py
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
from typing import IO

import numpy as np

from ..core import Serialisable
from ..core.algorithm import MatrixAlgorithm
from ..core.matrix import Matrix


class Log(MatrixAlgorithm, Serialisable):
    def __init__(self):
        super().__init__()

        self._base: float = np.e
        self.offset: float = 1.0

    def get_base(self) -> float:
        return self._base

    def set_base(self, value: float):
        if value == 1.0 or value <= 0.0:
            raise ValueError(f"Logarithmic base must not be 1, 0 or negative (got {value})")

        self._base = value

    base = property(get_base, set_base)

    def _do_transform(self, data: Matrix) -> Matrix:
        # Apply the offset
        result = data.add(self.offset)

        # Log is undefined for zero and negative values
        if result.any(lambda v: v <= 0):
            raise ValueError("Logarithm is undefined for negative/zero values")

        return result.log(in_place=True).multiply(self._base_conversion_factor(), in_place=True)

    def _do_inverse_transform(self, data: Matrix) -> Matrix:
        # Undo the logarithm
        result = data.divide(self._base_conversion_factor()).exp(in_place=True)

        return result.subtract(self.offset)

    def _base_conversion_factor(self) -> float:
        """
        Calculates the conversion factor for converting from the natural logarithm
        to logarithms of our base.

        :return:    The conversion factor, 1 / ln(base).
        """
        return 1.0 / np.log(self.base) if self.base != np.e else 1.0

    def serialise_state(self, stream: IO[bytes]):
        # Serialise out our base, base-conversion factor and offset
        stream.write(self.serialise_to_bytes(self.base))
        stream.write(self.serialise_to_bytes(self._base_conversion_factor()))
        stream.write(self.serialise_to_bytes(self.offset))
