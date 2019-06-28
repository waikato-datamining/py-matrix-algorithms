#  _AbstractPLSTest.py
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
import os
from typing import TypeVar, List, Optional

from ...test import AbstractRegressionTest
from ...test.misc import Test, Tags, TestDataset
from ....algorithm.pls import AbstractPLS
from ....core.matrix import Matrix

T = TypeVar('T', bound=AbstractPLS)


class AbstractPLSTest(AbstractRegressionTest[T]):
    @Test
    def check_transformed_num_components(self):
        X: Matrix = self.input_data[0]
        Y: Matrix = self.input_data[1]

        for i in range(1, 5):
            self.subject.num_components = i
            self.subject.initialize(X, Y)
            transform: Matrix = self.subject.transform(X)
            self.assertEqual(i, transform.num_columns())

            # Reset
            self.subject = self.instantiate_subject()

    @staticmethod
    def merge_if_not_empty(tag_1: str, tag_2: str) -> str:
        if tag_1 != '' and tag_2 != '':
            return os.sep.join((tag_1, tag_2))
        else:
            return tag_1 + tag_2

    def setup_regressions(self, subject: AbstractPLS, input_data: List[Matrix]):
        # Extract data
        X: Matrix = input_data[0]
        y: Matrix = input_data[1]

        # Initialise PLS
        results: Optional[str] = subject.initialize(X, y)
        if results is not None:
            self.fail('Algorithm#initialize failed with result: ' + results)

        self.add_default_pls_matrices(subject, X)

    def add_default_pls_matrices(self, algorithm: AbstractPLS, x: Matrix, sub_tag: str = ''):
        """
        Adds default PLS matrices, that is predictions, transformations, loadings and
        model parameter matrices.
        """
        self.add_predictions(algorithm, x, sub_tag)
        self.add_transformation(algorithm, x, sub_tag)
        self.add_loadings(algorithm, sub_tag)
        self.add_matrices(algorithm, sub_tag)

    def add_transformation(self, algorithm: AbstractPLS, x: Matrix, sub_tag: str = ''):
        """
        Add transformation to the regression group.
        """
        self.add_regression(self.merge_if_not_empty(sub_tag, Tags.TRANSFORM), algorithm.transform(x))

    def add_predictions(self, algorithm: AbstractPLS, x: Matrix, sub_tag: str = ''):
        """
        Add predictions to the regression group.
        """
        # Add predictions
        if algorithm.can_predict():
            preds: Matrix = algorithm.predict(x)
            self.add_regression(self.merge_if_not_empty(sub_tag, Tags.PREDICTIONS), preds)

    def add_loadings(self, algorithm: AbstractPLS, sub_tag: str = ''):
        """
        Add loadings to the regression group.
        """
        # Add loadings
        if algorithm.has_loadings():
            self.add_regression(self.merge_if_not_empty(sub_tag, Tags.LOADINGS), algorithm.get_loadings())

    def add_matrices(self, algorithm: AbstractPLS, sub_tag: str = ''):
        """
        Add model matrices to the regression group.
        """
        # Add matrices
        for matrix_name in algorithm.get_matrix_names():
            tag: str = Tags.MATRIX + '-' + matrix_name
            self.add_regression(self.merge_if_not_empty(sub_tag, tag), algorithm.get_matrix(matrix_name))

    def get_datasets(self) -> List[TestDataset]:
        return [TestDataset.BOLTS, TestDataset.BOLTS_RESPONSE]
