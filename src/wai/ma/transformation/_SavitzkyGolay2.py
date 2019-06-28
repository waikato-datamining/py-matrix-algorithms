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
        self.num_points: int = 3

        # Delete num_points_left/right
        del self.num_points_left
        del self.num_points_right

    def __getattribute__(self, item):
        # Alias num_points_left/right to num_points
        if item in {'num_points_left', 'num_points_right'}:
            return self.num_points

        return super().__getattribute__(item)

    @staticmethod
    def validate_num_points(value: int):
        if value < 0:
            raise ValueError('num_points must be at least 0')
