#  _MatrixRegression.py
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
from typing import Tuple, List

from ....core.matrix import Matrix, helper
from ._AbstractRegression import AbstractRegression


class MatrixRegression(AbstractRegression):
    """
    Matrix regression implementation.
    """
    def __init__(self, path: str, actual: Matrix):
        super().__init__(path, actual)

    def check_equals(self, expected: Matrix, actual: Matrix):
        # Check for shape
        if not expected.same_shape_as(actual):
            self.fail('Shapes of the expected and actual matrices do not match.\n'
                      + 'Expected shape: ' + expected.shape_string() +'\n'
                      + 'Actual shape: ' + actual.shape_string())

        # Check for values
        diff: Matrix = expected.sub(actual).abs()
        which = diff.which(lambda v: v > AbstractRegression.EPSILON)
        if len(which) > 0:
            self.fail('Regression ' + self.get_path() + ' failed.\n'
                      + 'Absolute differences:\n'
                      + self.indices_to_string(which, diff))

    def read_expected(self, path: str):
        return helper.read(path, False, ',')

    def write_expected(self, path: str, expected):
        helper.write(expected, path, False, ',', 10, False)

    def get_filename_extension(self) -> str:
        return 'csv'

    def indices_to_string(self, which: List[Tuple[int, int]], diff: Matrix) -> str:
        string = '(<row>, <column>): <absolute difference>\n'
        string += '\n'.join(('({i}, {j}): {diff}'.format_map({'i': i, 'j': j, 'diff': diff.get(i, j)})
                            for i, j in which))
        return string
