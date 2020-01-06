#  _MatrixTest.py
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

from wai.ma.core.matrix import Matrix, factory
from wai.test.decorators import RegressionTest, Test

from ...test import TestDataset, Tags, AbstractMatrixAlgorithmTest


class MatrixTest(AbstractMatrixAlgorithmTest):
    @classmethod
    def subject_type(cls):
        return Matrix

    @classmethod
    def instantiate_subject(cls) -> Matrix:
        return factory.randn(7, 7, seed=2)

    @classmethod
    def get_datasets(cls) -> Tuple[TestDataset]:
        return TestDataset.BOLTS,

    @RegressionTest
    def transpose(self, subject: Matrix, bolts: Matrix):
        return self.standard_regression(subject.transpose())

    @RegressionTest
    def get_eigenvectors(self, subject: Matrix, bolts: Matrix):
        return self.standard_regression(subject.get_eigenvectors())

    @RegressionTest
    def get_eigenvalues(self, subject: Matrix, bolts: Matrix):
        return self.standard_regression(subject.get_eigenvalues())

    @RegressionTest
    def svd_S(self, subject: Matrix, bolts: Matrix):
        return self.standard_regression(subject.svd_S())

    @RegressionTest
    def svd_U(self, subject: Matrix, bolts: Matrix):
        return self.standard_regression(subject.svd_U())

    @RegressionTest
    def svd_V(self, subject: Matrix, bolts: Matrix):
        return self.standard_regression(subject.svd_V())

    @RegressionTest
    def norm2(self, subject: Matrix, bolts: Matrix):
        return self.standard_regression(bolts.norm2())

    @RegressionTest
    def norm1(self, subject: Matrix, bolts: Matrix):
        return self.standard_regression(bolts.norm1())

    @Test
    def column_is_sub_matrix(self, subject: Matrix, bolts: Matrix):
        for i in range(subject.num_columns()):
            column = subject.get_column(i)
            sub_matrix = subject.get_sub_matrix((0, subject.num_rows()),
                                                (i, i + 1))
            self.assertEqual(column, sub_matrix)

    def standard_regression(self, subject: Matrix, *resources):
        # Only regression testing the final configuration of the matrix
        return {
            Tags.MATRIX: subject
        }
