#  _SavitzkyGolayTest.py
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
from typing import List, Optional

from ..test import AbstractRegressionTest
from ..test.misc import TestRegression, Tags, TestDataset
from wai.ma.core.matrix import Matrix
from wai.ma.transformation import SavitzkyGolay


class SavitzkyGolayTest(AbstractRegressionTest[SavitzkyGolay]):
    @TestRegression
    def coefficients_3_3_2_0(self):
        self.coefficient_helper(3, 3, 2, 0)

    @TestRegression
    def coefficients_3_3_2_1(self):
        self.coefficient_helper(3, 3, 2, 1)

    @TestRegression
    def coefficients_3_3_2_2(self):
        self.coefficient_helper(3, 3, 2, 2)

    @TestRegression
    def coefficients_0_3_2_1(self):
        self.coefficient_helper(0, 3, 2, 1)

    @TestRegression
    def coefficients_3_0_2_1(self):
        self.coefficient_helper(3, 0, 2, 1)

    def setup_regressions(self, subject: SavitzkyGolay, input_data: List[Matrix]):
        X: Matrix = input_data[0]

        transform: Matrix = subject.transform(X)

        self.add_regression(Tags.TRANSFORM, transform)
        self.add_regression("coefficients", subject.coefficients)

    def get_datasets(self) -> List[TestDataset]:
        return [TestDataset.BOLTS]

    def instantiate_subject(self) -> SavitzkyGolay:
        return SavitzkyGolay()

    def coefficient_helper(self,
                           num_points_left: Optional[int] = None,
                           num_points_right: Optional[int] = None,
                           polynomial_order: Optional[int] = None,
                           derivative_order: Optional[int] = None):
        if polynomial_order is not None:
            self.subject.polynomial_order = polynomial_order
        if derivative_order is not None:
            self.subject.derivative_order = derivative_order
        if num_points_left is not None:
            self.subject.num_points_left = num_points_left
        if num_points_right is not None:
            self.subject.num_points_right = num_points_right
