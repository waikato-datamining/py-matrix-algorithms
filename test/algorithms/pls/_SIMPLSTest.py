#  _SIMPLSTest.py
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
from wai.test.decorators import RegressionTest

from wai.ma.algorithm.pls import SIMPLS
from wai.ma.core.matrix import Matrix

from ._AbstractPLSTest import AbstractPLSTest


class SIMPLSTest(AbstractPLSTest):
    @classmethod
    def subject_type(cls):
        return SIMPLS

    @RegressionTest
    def num_coefficients_1(self, subject: SIMPLS, *resources: Matrix):
        subject.num_coefficients = 1
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def num_coefficients_2(self, subject: SIMPLS, *resources: Matrix):
        subject.num_coefficients = 2
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def num_coefficients_3(self, subject: SIMPLS, *resources: Matrix):
        subject.num_coefficients = 3
        return self.standard_regression(subject, *resources)

    def standard_regression(self, subject: SIMPLS, *resources: Matrix):
        regression = super().standard_regression(subject, *resources)
        regression.update({
            "serialised": subject.get_serialised_state()
        })
        return regression
