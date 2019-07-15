#  _PassThroughTest.py
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
from ._AbstractTransformationTest import AbstractTransformationTest
from ..test.misc import Test
from wai.ma.core.matrix import Matrix
from wai.ma.transformation import PassThrough


class PassThroughTest(AbstractTransformationTest[PassThrough]):
    @Test
    def result_unchanged(self):
        X: Matrix = self.input_data[0]
        transform: Matrix = self.subject.transform(X)

        is_equal: bool = X.sub(transform).abs().all(lambda v: v < 1e-7)
        self.assertTrue(is_equal)

    def instantiate_subject(self) -> PassThrough:
        return PassThrough()
