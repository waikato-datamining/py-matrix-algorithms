#  _MatrixAlgorithmTest.py
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
from wai.ma.core.algorithm import MatrixAlgorithm, SupervisedMatrixAlgorithm, UnsupervisedMatrixAlgorithm

from ..test import AbstractMatrixAlgorithmTest, TestDataset, Tags


class MatrixAlgorithmTest(AbstractMatrixAlgorithmTest, ABC):
    @Test
    def check_inv_transform_eq_input(self, subject: MatrixAlgorithm, *resources: Matrix):
        bolts, bolts_response = resources

        self.configure(subject, *resources)

        if subject.is_non_invertible():
            return

        transform: Matrix = subject.transform(bolts)
        inverse_transform: Matrix = subject.inverse_transform(transform)

        # Check if input == inverse_transform
        is_equal: bool = bolts.subtract(inverse_transform).abs().all(lambda v: v < 1e-7)
        self.assertTrue(is_equal)

    def standard_regression(self, subject: MatrixAlgorithm, *resources: Matrix):
        bolts, bolts_response = resources

        self.configure(subject, *resources)

        transform: Matrix = subject.transform(bolts)

        result = {
            Tags.TRANSFORM: transform
        }

        if not subject.is_non_invertible():
            result.update({
                Tags.INVERSE_TRANSFORM: subject.inverse_transform(transform)
            })

        return result

    @classmethod
    def get_datasets(cls) -> Tuple[TestDataset, TestDataset]:
        return TestDataset.BOLTS, TestDataset.BOLTS_RESPONSE

    def configure(self, subject: MatrixAlgorithm, *resources: Matrix):
        """
        Configures the subject algorithm on the input data,
        if it is required.

        :param subject:     The subject algorithm.
        :param resources:   The input data.
        """
        # Unpack the data matrices
        bolts, bolts_response = resources

        # Perform configuration for the type of algorithm
        if isinstance(subject, SupervisedMatrixAlgorithm):
            subject.configure(bolts, bolts_response)
        elif isinstance(subject, UnsupervisedMatrixAlgorithm):
            subject.configure(bolts)
