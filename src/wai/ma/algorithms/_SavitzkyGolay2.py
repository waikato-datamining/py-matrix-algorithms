#  _SavitzkyGolay2.py
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

from ._SavitzkyGolay import SavitzkyGolay


class SavitzkyGolay2(SavitzkyGolay):
    def __init__(self):
        super().__init__()

        # Defaults
        self._num_points: int = 3

    def get_num_points(self) -> int:
        return self._num_points

    def set_num_points(self, value: int):
        if value < 0:
            raise ValueError(f"num_points must be at least 0, got {value}")

        self._num_points = self._num_points_left = self._num_points_right = value

    num_points = property(get_num_points, set_num_points)

    def set_num_points_left(self, value: int):
        self.set_num_points(value)

    def set_num_points_right(self, value: int):
        self.set_num_points(value)
