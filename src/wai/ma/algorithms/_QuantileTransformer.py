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

from ..core.algorithm import UnsupervisedMatrixAlgorithm
from ..core.matrix import Matrix


class QuantileTransformer(UnsupervisedMatrixAlgorithm):
    # The output distribution options
    UNIFORM_OUTPUT_DISTRIBUTION = "uniform"
    NORMAL_OUTPUT_DISTRIBUTION = "normal"

    def __init__(self):
        super().__init__()

        self.n_quantiles: int = 1000
        self._output_distribution: str = QuantileTransformer.UNIFORM_OUTPUT_DISTRIBUTION
        self.ignore_implicit_zeroes: bool = False
        self.subsample: int = 100000
        self.random_state: Optional[int] = None

        self._intern: Optional[Intern] = None

    def get_output_distribution(self) -> str:
        return self._output_distribution

    def set_output_distribution(self, value: str):
        available_distributions = {QuantileTransformer.UNIFORM_OUTPUT_DISTRIBUTION,
                                   QuantileTransformer.NORMAL_OUTPUT_DISTRIBUTION}
        if value not in available_distributions:
            raise ValueError(f"Output distribution was {value} but must be one of: " +
                             ", ".join(available_distributions))

        self._output_distribution = value
        self.reset()

    output_distribution = property(get_output_distribution, set_output_distribution)

    def _do_reset(self):
        super()._do_reset()
        self._intern = None

    def _do_configure(self, data: Matrix):
        self._intern = Intern(n_quantiles=self.n_quantiles,
                              output_distribution=self._output_distribution,
                              ignore_implicit_zeros=self.ignore_implicit_zeroes,
                              subsample=self.subsample,
                              random_state=self.random_state)
        self._intern.fit(data._data)

    def _do_transform(self, data: Matrix) -> Matrix:
        return Matrix(self._intern.transform(data._data))

    def _do_inverse_transform(self, data: Matrix) -> Matrix:
        return Matrix(self._intern.inverse_transform(data._data))
