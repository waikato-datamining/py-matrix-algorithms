#  _QuantileTransformer.py
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

from sklearn.preprocessing import QuantileTransformer as Intern

from ..core.matrix import Matrix
from ._AbstractTransformation import AbstractTransformation


class QuantileTransformer(AbstractTransformation):
    # The output distribution options
    OUTPUT_DISTRIBUTIONS = ("uniform", "normal")

    def __init__(self):
        super().__init__()

        self.n_quantiles: int = 1000
        self.output_distribution: str = QuantileTransformer.OUTPUT_DISTRIBUTIONS[0]
        self.ignore_implicit_zeroes: bool = False
        self.subsample: int = 100000
        self.random_state: Optional[int] = None

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

    def validate_output_distribution(self, selection: str):
        if selection not in QuantileTransformer.OUTPUT_DISTRIBUTIONS:
            raise ValueError(f"Output distribution was {selection} but must be one of: " +
                             ", ".join(QuantileTransformer.OUTPUT_DISTRIBUTIONS))

    def reset(self):
        super().reset()
        self._intern = None

    def configure(self, data: Matrix):
        self._intern = Intern(n_quantiles=self.n_quantiles,
                              output_distribution=self.output_distribution,
                              ignore_implicit_zeros=self.ignore_implicit_zeroes,
                              subsample=self.subsample,
                              random_state=self.random_state)
        self._intern.fit(data.data)
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        return Matrix(self._intern.transform(data.data))

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        return Matrix(self._intern.inverse_transform(data.data))
