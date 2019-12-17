#  _CCAFilterTest.py
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
from wai.ma.algorithms import CCAFilter
from wai.ma.core.matrix import Matrix

from ..test import Tags
from ._MatrixAlgorithmTest import MatrixAlgorithmTest


class CCAFilterTest(MatrixAlgorithmTest):
    @classmethod
    def subject_type(cls):
        return CCAFilter

    @RegressionTest
    def lambda_X_10(self, subject: CCAFilter, *resources: Matrix):
        subject.lambda_X = 10
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def lambda_Y_10(self, subject: CCAFilter, *resources: Matrix):
        subject.lambda_Y = 10
        return self.standard_regression(subject, *resources)

    def standard_regression(self, subject: CCAFilter, *resources: Matrix):
        regressions = super().standard_regression(subject, *resources)

        regressions.update({
            Tags.PROJECTION + '-X': subject.get_projection_matrix_X(),
            Tags.PROJECTION + '-Y': subject.get_projection_matrix_Y()
        })

        return regressions

