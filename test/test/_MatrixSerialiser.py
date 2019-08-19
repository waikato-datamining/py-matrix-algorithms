#  _MatrixSerialiser.py
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
from typing import IO, Optional, List, Tuple

from wai.test.serialisation import RegressionSerialiser
from wai.ma.core.matrix import Matrix, helper

from ._epsilon import EPSILON


class MatrixSerialiser(RegressionSerialiser[Matrix]):
    @classmethod
    def binary(cls) -> bool:
        return False

    @classmethod
    def extension(cls) -> str:
        return "csv"

    @classmethod
    def serialise(cls, result: Matrix, file: IO[str]):
        helper.write(result, file, False, ',', 10, False)

    @classmethod
    def deserialise(cls, file: IO[str]) -> Matrix:
        return helper.read(file, False, ",")

    @classmethod
    def compare(cls, result: Matrix, reference: Matrix) -> Optional[str]:
        # Check for shape
        if not reference.same_shape_as(result):
            return 'Shapes of the expected and actual matrices do not match.\n' + \
                   'Expected shape: ' + reference.shape_string() + '\n' + \
                   'Actual shape: ' + result.shape_string()

        # Check for values
        diff: Matrix = reference.sub(result).abs()
        which = diff.which(lambda v: v > EPSILON)
        if len(which) > 0:
            return 'Values in the expected and actual matrices do not match.\n' + \
                   'Absolute differences:\n' + \
                   cls.indices_to_string(which, diff)

    @classmethod
    def indices_to_string(cls, which: List[Tuple[int, int]], diff: Matrix) -> str:
        string = '(<row>, <column>): <absolute difference>\n'
        string += '\n'.join(('({i}, {j}): {diff}'.format_map({'i': i, 'j': j, 'diff': diff.get(i, j)})
                            for i, j in which))
        return string
