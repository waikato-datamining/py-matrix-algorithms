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
from typing import List

from wai.ma.algorithm import PCA
from wai.ma.core.matrix import Matrix
from ..test import AbstractRegressionTest
from ..test.misc import TestRegression, Tags, TestDataset


class PCATest(AbstractRegressionTest[PCA]):
    @TestRegression
    def center(self):
        self.subject.center = True

    @TestRegression
    def max_cols_3(self):
        self.subject.max_columns = 3

    def setup_regressions(self, subject: PCA, input_data: List[Matrix]):
        # Get input
        X: Matrix = self.input_data[0]

        # Get matrices
        transformed: Matrix = subject.transform(X)
        loadings: Matrix = subject.loadings
        scores: Matrix = subject.scores

        # Add regressions
        self.add_regression(Tags.TRANSFORM, transformed)
        self.add_regression(Tags.LOADINGS, loadings)
        self.add_regression(Tags.SCORES, scores)

    def get_datasets(self) -> List[TestDataset]:
        return [TestDataset.BOLTS]

    def instantiate_subject(self) -> PCA:
        return PCA()
