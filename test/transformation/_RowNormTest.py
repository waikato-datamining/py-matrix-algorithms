#  RowNormTest.py
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
from wai.test.decorators import Test, ExpectedFailure
from wai.ma.core import real
from wai.ma.core.matrix import Matrix, helper
from wai.ma.transformation import RowNorm

from ..transformation import AbstractTransformationTest


class RowNormTest(AbstractTransformationTest):
    @classmethod
    def subject_type(cls):
        return RowNorm

    @Test
    def check_zero_variance(self, subject: RowNorm, bolts: Matrix):
        transform: Matrix = subject.transform(bolts)

        actual_mean: real = transform.mean()
        expected_mean: real = real(0.0)
        expected_std: real = real(1.0)

        self.assertAlmostEqual(expected_mean, actual_mean, delta=1e-7)

        for i in range(transform.num_rows()):
            actual_std: real = helper.stdev(transform, i, column=False)
            self.assertAlmostEqual(expected_std, actual_std, delta=1e-7)

    @ExpectedFailure
    def check_inv_transform_eq_input(self):
        """
        This test will fail for RowNorm because it reconfigures on every input.
        """
        super().check_inv_transform_eq_input()
