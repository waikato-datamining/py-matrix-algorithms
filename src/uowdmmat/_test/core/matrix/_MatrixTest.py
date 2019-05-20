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
from typing import List

from ...test.misc import TestDataset, Test, Tags, TestRegression
from ...test import AbstractRegressionTest
from ....core.matrix import Matrix, factory


class MatrixTest(AbstractRegressionTest[Matrix]):
    @TestRegression
    def transpose(self):
        self.subject = self.subject.transpose()

    @TestRegression
    def get_eigenvectors(self):
        self.subject = self.subject.get_eigenvectors()

    @TestRegression
    def get_eigenvalues(self):
        self.subject = self.subject.get_eigenvalues()

    def setup_regressions(self, subject: Matrix, input_data: List[Matrix]):
        # Only regression testing the final configuration of the matrix
        self.add_regression(Tags.MATRIX, subject)

    def get_datasets(self) -> List[TestDataset]:
        return []

    def instantiate_subject(self) -> Matrix:
        return factory.randn(7, 7, seed=2)

