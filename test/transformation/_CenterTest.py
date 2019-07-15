#  _CenterTest.py
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
from ..test.misc import Test
from ..transformation import AbstractTransformationTest
from wai.ma.core import real
from wai.ma.core.matrix import Matrix
from wai.ma.transformation import Center


class CenterTest(AbstractTransformationTest[Center]):
    @Test
    def mean_is_zero(self):
        X: Matrix = self.input_data[0]
        transform: Matrix = self.subject.transform(X)
        actual: real = transform.mean()
        expected: real = real(0.0)
        self.assertAlmostEqual(actual, expected, delta=1e-7)

    def instantiate_subject(self) -> Center:
        return Center()