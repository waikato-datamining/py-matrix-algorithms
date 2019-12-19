#  _EquidistanceTest.py
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
from wai.test.decorators import RegressionTest, Test
from wai.ma.core.matrix import Matrix
from wai.ma.algorithms import Equidistance

from ._MatrixAlgorithmTest import MatrixAlgorithmTest


class EquidistanceTest(MatrixAlgorithmTest):
    @classmethod
    def subject_type(cls):
        return Equidistance

    @RegressionTest
    def filter_3(self, subject: Equidistance, *resources: Matrix):
        subject.num_samples = 3
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def filter_4(self, subject: Equidistance, *resources: Matrix):
        subject.num_samples = 4
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def filter_5(self, subject: Equidistance, *resources: Matrix):
        subject.num_samples = 5
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def filter_6(self, subject: Equidistance, *resources: Matrix):
        subject.num_samples = 6
        return self.standard_regression(subject, *resources)

    @Test
    def identity(self, subject: Equidistance, *resources: Matrix):
        bolts, bolts_response = resources

        subject.num_samples = bolts.num_columns()
        bolts_filtered = subject.transform(bolts)
        self.assertMatricesEqual(bolts, bolts_filtered)
