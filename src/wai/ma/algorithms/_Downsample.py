#  _Downsample.py
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

from ..core.algorithm import MatrixAlgorithm
from ..core.matrix import Matrix


class Downsample(MatrixAlgorithm):
    """
    Filter which gets every Nth row from a matrix, starting at a given index.
    """
    def __init__(self):
        super().__init__()

        self._start_index: int = 0  # The index to start sampling from
        self._step: int = 1  # The step-size between samples

    def get_start_index(self) -> int:
        return self._start_index

    def set_start_index(self, value: int):
        if value < 0:
            raise ValueError(f"Start index must be at least 0, was {value}")

        self._start_index = value

    start_index = property(get_start_index, set_start_index)

    def get_step(self) -> int:
        return self._step

    def set_step(self, value: int):
        if value < 1:
            raise ValueError(f"Step must be at least 1, was {value}")

        self._step = value

    step = property(get_step, set_step)

    def _do_transform(self, predictors: Matrix) -> Matrix:
        if self._start_index >= predictors.num_rows():
            raise IndexError(f"Start index ({self._start_index}) is beyond the end "
                             f"of the given matrix (rows = {predictors.num_rows()})")

        rows = [i for i in range(self.start_index, predictors.num_rows(), self.step)]

        return predictors.get_rows(rows)

    def is_non_invertible(self) -> bool:
        return True
