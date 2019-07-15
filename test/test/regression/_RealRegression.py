#  _DoubleRegression.py
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
from wai.ma.core import real
from ._AbstractRegression import AbstractRegression


class RealRegression(AbstractRegression):
    """
    Real value regression implementation.
    """
    def __init__(self, path: str, actual: real):
        super().__init__(path, actual)

    def check_equals(self, expected, actual):
        self.assertAlmostEqual(expected, actual, delta=AbstractRegression.EPSILON)

    def read_expected(self, path: str):
        with open(path, 'r') as file:
            return real(file.readline())

    def write_expected(self, path: str, expected):
        with open(path, 'w') as file:
            file.writelines(str(expected))

    def get_filename_extension(self) -> str:
        return 'txt'
