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
from typing import Optional

from wai.test.decorators import RegressionTest, ExceptionTest
from wai.ma.core.matrix import Matrix
from wai.ma.transformation import SavitzkyGolay

from ..test import Tags
from ._AbstractTransformationTest import AbstractTransformationTest


class SavitzkyGolayTest(AbstractTransformationTest):
    @classmethod
    def subject_type(cls):
        return SavitzkyGolay

    @RegressionTest
    def coefficients_3_3_2_0(self, subject: SavitzkyGolay, bolts: Matrix):
        self.coefficient_helper(subject, 3, 3, 2, 0)
        return self.standard_regression(subject, bolts)

    @RegressionTest
    def coefficients_3_3_2_1(self, subject: SavitzkyGolay, bolts: Matrix):
        self.coefficient_helper(subject, 3, 3, 2, 1)
        return self.standard_regression(subject, bolts)

    @RegressionTest
    def coefficients_3_3_2_2(self, subject: SavitzkyGolay, bolts: Matrix):
        self.coefficient_helper(subject, 3, 3, 2, 2)
        return self.standard_regression(subject, bolts)

    @RegressionTest
    def coefficients_0_3_2_1(self, subject: SavitzkyGolay, bolts: Matrix):
        self.coefficient_helper(subject, 0, 3, 2, 1)
        return self.standard_regression(subject, bolts)

    @RegressionTest
    def coefficients_3_0_2_1(self, subject: SavitzkyGolay, bolts: Matrix):
        self.coefficient_helper(subject, 3, 0, 2, 1)
        return self.standard_regression(subject, bolts)

    @ExceptionTest(NotImplementedError)
    def check_inv_transform_eq_input(self, subject: SavitzkyGolay, bolts: Matrix):
        """
        Savitzky-Golay doesn't have an inverse transform.
        """
        super().check_inv_transform_eq_input()

    def standard_regression(self, subject: SavitzkyGolay, *resources: Matrix):
        bolts, = resources

        transform: Matrix = subject.transform(bolts)

        return {
            Tags.TRANSFORM: transform,
            "coefficients": subject.coefficients,
            "serialised": subject.get_serialised_state()
        }

    @staticmethod
    def coefficient_helper(subject: SavitzkyGolay,
                           num_points_left: Optional[int] = None,
                           num_points_right: Optional[int] = None,
                           polynomial_order: Optional[int] = None,
                           derivative_order: Optional[int] = None):
        if polynomial_order is not None:
            subject.polynomial_order = polynomial_order
        if derivative_order is not None:
            subject.derivative_order = derivative_order
        if num_points_left is not None:
            subject.num_points_left = num_points_left
        if num_points_right is not None:
            subject.num_points_right = num_points_right
