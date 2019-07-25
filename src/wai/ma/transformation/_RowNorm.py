#  _RowNorm.py
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

from ..core import real
from ..core.matrix import Matrix
from ..core.matrix.helper import row_means, row_stdevs
from ._AbstractTransformation import AbstractTransformation


class RowNorm(AbstractTransformation):
    def __init__(self):
        super().__init__()
        self.means: Optional[Matrix] = None
        self.std_devs: Optional[Matrix] = None

    def reset(self):
        super().reset()
        self.means = None
        self.std_devs = None

    def configure(self, data: Matrix):
        self.means = row_means(data)
        self.std_devs = row_stdevs(data)
        # Make sure we don't do a divide by zero
        for i in range(self.std_devs.num_rows()):
            if self.std_devs.get(i, 0) == real(0):
                self.std_devs.set(i, 0, 1)

    def do_transform(self, data: Matrix) -> Matrix:
        result = data.copy()
        result.sub_by_vector_modify(self.means)
        result.div_by_vector_modify(self.std_devs)
        return result

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        result = data.copy()
        result.mul_by_vector_modify(self.std_devs)
        result.add_by_vector_modify(self.means)
        return result

    def get_means(self) -> Matrix:
        return self.means

    def get_std_devs(self) -> Matrix:
        return self.std_devs
