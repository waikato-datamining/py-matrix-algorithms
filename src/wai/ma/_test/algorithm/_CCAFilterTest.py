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
from typing import List

from ..test import AbstractRegressionTest, T
from ..test.misc import TestRegression, Tags, TestDataset
from ...algorithm import CCAFilter
from ...core.matrix import Matrix


class CCAFilterTest(AbstractRegressionTest[CCAFilter]):
    @TestRegression
    def lambda_X_10(self):
        self.subject.lambda_X = 10

    @TestRegression
    def lambda_Y_10(self):
        self.subject.lambda_Y = 10

    def setup_regressions(self, subject: T, input_data: List[Matrix]):
        X: Matrix = self.input_data[0]
        Y: Matrix = self.input_data[1]

        self.subject.initialize(X, Y)

        self.add_regression(Tags.TRANSFORM, self.subject.transform(X))
        self.add_regression(Tags.PROJECTION + '-X', self.subject.proj_X)
        self.add_regression(Tags.PROJECTION + '-Y', self.subject.proj_Y)

    def get_datasets(self) -> List[TestDataset]:
        return [TestDataset.BOLTS, TestDataset.BOLTS_RESPONSE]

    def instantiate_subject(self) -> CCAFilter:
        return CCAFilter()
