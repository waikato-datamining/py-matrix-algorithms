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
from ..core.matrix import Matrix
from ._AbstractTransformation import AbstractTransformation


class Log(AbstractTransformation, Serialisable):
    def __init__(self):
        super().__init__()

        self.base: float = np.e
        self.offset: float = 1.0

    def __setattr__(self, key, value):
        # Validate values
        validator_name = 'validate_' + key
        if hasattr(self, validator_name):
            validator = getattr(self, validator_name)
            validator(value)

            # If validation passes, reset
            self.reset()

        super().__setattr__(key, value)

    @staticmethod
    def validate_base(value: float):
        if value == 1.0 or value <= 0.0:
            raise ValueError(f"Logarithmic base must not be 1, 0 or negative (got {value})")

    def configure(self, data: Matrix):
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        # Copy the matrix data
        data: np.ndarray = data.data.copy()

        # Apply the offset
        data += self.offset

        # Log is undefined for zero and negative values
        undefined: np.ndarray = data <= 0
        if undefined.any():
            raise ValueError("Logarithm is undefined for negative/zero values")

        return Matrix(np.log(data) * self._base_conversion_factor())

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        # Copy the matrix data
        data: np.ndarray = data.data.copy()

        # Undo the logarithm
        data = np.exp(data / self._base_conversion_factor())

        # Undo the offset
        data -= self.offset

        return Matrix(data)

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
