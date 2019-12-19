#  _PCATest.py
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
from wai.ma.algorithms import PCA
from wai.ma.core.matrix import Matrix

from ..test import Tags
from ._MatrixAlgorithmTest import MatrixAlgorithmTest


class PCATest(MatrixAlgorithmTest):
    @classmethod
    def subject_type(cls):
        return PCA

    @RegressionTest
    def center(self, subject: PCA, *resources: Matrix):
        subject.center = True
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def max_cols_3(self, subject: PCA, *resources: Matrix):
        subject.max_columns = 3
        return self.standard_regression(subject, *resources)

    def standard_regression(self, subject: PCA, *resources: Matrix):
        # Get input
        bolts, bolts_response = resources

        # Get matrices
        transformed: Matrix = subject.transform(bolts)
        loadings: Matrix = subject.loadings
        scores: Matrix = subject.scores

        # Add regressions
        return {
            Tags.TRANSFORM: transformed,
            Tags.LOADINGS: loadings,
            Tags.SCORES: scores,
        }
