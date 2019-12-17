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

from wai.ma.core.matrix import Matrix, factory, Axis
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
        return self.standard_regression(subject.head(subject.num_columns()).get_eigenvectors())

    @RegressionTest
    def get_eigenvalues(self, subject: Matrix, bolts: Matrix):
        return self.standard_regression(subject.get_eigenvalues())

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

    @Test
    def aggregations_are_right_shape(self, subject: Matrix, bolts: Matrix):
        """
        Tests that the row and column aggregations are the correct shape.

        :param subject:     The test matrix.
        :param bolts:       The bolts matrix.
        """
        for aggregation in (bolts.maximum,
                            bolts.minimum,
                            bolts.median,
                            bolts.mean,
                            bolts.standard_deviation,
                            bolts.total,
                            bolts.norm1):
            with self.subTest(aggregation.__name__):
                bolts_columns = bolts.mean(Axis.COLUMNS)
                self.assertTrue(bolts_columns.is_row_vector())
                self.assertEqual(bolts_columns.num_columns(), 7)

                bolts_rows = bolts.mean(Axis.ROWS)
                self.assertTrue(bolts_rows.is_column_vector())
                self.assertEqual(bolts_rows.num_rows(), 40)

    def standard_regression(self, subject: Matrix, *resources):
        # Only regression testing the final configuration of the matrix
        return {
            Tags.MATRIX: subject
        }
