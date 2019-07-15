#  _AbstractTransformationTest.py
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
from ..test.misc import Test, Tags, TestDataset
from wai.ma.core.matrix import Matrix


class AbstractTransformationTest(AbstractRegressionTest[T]):
    """
    Abstract tranformation test. Regression for transform and inverse-transform.
    """
    @Test
    def check_inv_transform_eq_input(self):
        input: Matrix = self.input_data[0]
        transform: Matrix = self.subject.transform(input)
        inverse_transform: Matrix = self.subject.inverse_transform(transform)

        # Check if input == inverse_transform
        is_equal: bool = input.sub(inverse_transform).abs().all(lambda v: v < 1e-7)
        self.assertTrue(is_equal)

    def setup_regressions(self, subject: T, input_data: List[Matrix]):
        X: Matrix = input_data[0]
        subject.configure(X)

        transform: Matrix = subject.transform(X)
        inverse_transform: Matrix = subject.inverse_transform(transform)

        self.add_regression(Tags.TRANSFORM, transform)
        self.add_regression(Tags.INVERSE_TRANSFORM, inverse_transform)

    def get_datasets(self) -> List[TestDataset]:
        return [TestDataset.BOLTS]