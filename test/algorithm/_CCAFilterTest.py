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
from typing import Tuple

from wai.test.decorators import RegressionTest
from wai.ma.algorithm import CCAFilter
from wai.ma.core.matrix import Matrix

from ..test import AbstractMatrixAlgorithmTest, Tags, TestDataset


class CCAFilterTest(AbstractMatrixAlgorithmTest):
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
        bolts, bolts_response = resources

        subject.initialize(bolts, bolts_response)

        return {
            Tags.TRANSFORM: subject.transform(bolts),
            Tags.PROJECTION + '-X': subject.proj_X,
            Tags.PROJECTION + '-Y': subject.proj_Y
        }

    @classmethod
    def get_datasets(cls) -> Tuple[TestDataset, TestDataset]:
        return TestDataset.BOLTS, TestDataset.BOLTS_RESPONSE

