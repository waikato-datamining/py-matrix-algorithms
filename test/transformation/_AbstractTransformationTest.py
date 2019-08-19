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
from abc import ABC
from typing import Tuple

from wai.test.decorators import Test
from wai.ma.core.matrix import Matrix
from wai.ma.transformation import AbstractTransformation

from ..test import AbstractMatrixAlgorithmTest, TestDataset, Tags


class AbstractTransformationTest(AbstractMatrixAlgorithmTest, ABC):
    """
    Abstract tranformation test. Regression for transform and inverse-transform.
    """
    @Test
    def check_inv_transform_eq_input(self, subject: AbstractTransformation, bolts: Matrix):
        transform: Matrix = subject.transform(bolts)
        inverse_transform: Matrix = subject.inverse_transform(transform)

        # Check if input == inverse_transform
        is_equal: bool = bolts.sub(inverse_transform).abs().all(lambda v: v < 1e-7)
        self.assertTrue(is_equal)

    def standard_regression(self, subject: AbstractTransformation, *resources: Matrix):
        bolts, = resources

        subject.configure(bolts)

        transform: Matrix = subject.transform(bolts)
        inverse_transform: Matrix = subject.inverse_transform(transform)

        return {
            Tags.TRANSFORM: transform,
            Tags.INVERSE_TRANSFORM: inverse_transform
        }

    @classmethod
    def get_datasets(cls) -> Tuple[TestDataset]:
        return TestDataset.BOLTS,
