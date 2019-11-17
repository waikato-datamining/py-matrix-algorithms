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

from ..core.matrix import Matrix
from ._AbstractTransformation import AbstractTransformation


class RobustScaler(AbstractTransformation):
    # The output distribution options
    OUTPUT_DISTRIBUTIONS = ("uniform", "normal")

    def __init__(self):
        super().__init__()

        self.with_centering: bool = True
        self.with_scaling: bool = True
        self.quantile_range: Tuple[float, float] = (25.0, 75.0)

        self._intern: Optional[Intern] = None

    def __setattr__(self, key, value):
        # Validate values
        validator_name = 'validate_' + key
        if hasattr(self, validator_name):
            validator = getattr(self, validator_name)
            validator(value)

            # If validation passes, reset
            self.reset()

        super().__setattr__(key, value)

    def validate_quantile_range(self, selection: Tuple[float, float]):
        if not (0.0 < selection[0] < selection[1] < 100.0):
            raise ValueError(f"Quantile range must be 0.0 < q_min < q_max < 100.0. "
                             f"Got q_min={selection[0]}, q_max={selection[1]}")

    def reset(self):
        super().reset()
        self._intern = None

    def configure(self, data: Matrix):
        self._intern = Intern(with_centering=self.with_centering,
                              with_scaling=self.with_scaling,
                              quantile_range=self.quantile_range)
        self._intern.fit(data.data)
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        return Matrix(self._intern.transform(data.data))

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        return Matrix(self._intern.inverse_transform(data.data))
