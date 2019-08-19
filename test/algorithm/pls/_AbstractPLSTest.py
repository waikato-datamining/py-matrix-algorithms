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
from abc import ABC
from typing import Optional, Tuple

from wai.test.decorators import Test
from wai.ma.algorithm.pls import AbstractPLS
from wai.ma.core.matrix import Matrix

from ...test import AbstractMatrixAlgorithmTest, TestDataset, Tags


class AbstractPLSTest(AbstractMatrixAlgorithmTest, ABC):
    @Test
    def check_transformed_num_components(self, subject: AbstractPLS, bolts: Matrix, bolts_response: Matrix):
        for i in range(1, 5):
            subject.num_components = i
            subject.initialize(bolts, bolts_response)
            transform: Matrix = subject.transform(bolts)
            self.assertEqual(i, transform.num_columns())

            # Reset
            subject = self.instantiate_subject()

    @staticmethod
    def merge_if_not_empty(tag_1: str, tag_2: str) -> str:
        if tag_1 != '' and tag_2 != '':
            return os.sep.join((tag_1, tag_2))
        else:
            return tag_1 + tag_2

    def standard_regression(self, subject: AbstractPLS, *resources: Matrix):
        # Extract data
        bolts, bolts_response = resources

        # Initialise PLS
        results: Optional[str] = subject.initialize(bolts, bolts_response)
        if results is not None:
            self.fail('Algorithm#initialize failed with result: ' + results)

        return self.add_default_pls_matrices(subject, bolts)

    def add_default_pls_matrices(self, algorithm: AbstractPLS, x: Matrix, sub_tag: str = ''):
        """
        Adds default PLS matrices, that is predictions, transformations, loadings and
        model parameter matrices.
        """
        result = {}

        result.update(self.add_predictions(algorithm, x, sub_tag))
        result.update(self.add_transformation(algorithm, x, sub_tag))
        result.update(self.add_loadings(algorithm, sub_tag))
        result.update(self.add_matrices(algorithm, sub_tag))

        return result

    def add_transformation(self, algorithm: AbstractPLS, x: Matrix, sub_tag: str = ''):
        """
        Add transformation to the regression group.
        """
        return {self.merge_if_not_empty(sub_tag, Tags.TRANSFORM): algorithm.transform(x)}

    def add_predictions(self, algorithm: AbstractPLS, x: Matrix, sub_tag: str = ''):
        """
        Add predictions to the regression group.
        """
        # Add predictions
        if algorithm.can_predict():
            preds: Matrix = algorithm.predict(x)
            return {self.merge_if_not_empty(sub_tag, Tags.PREDICTIONS): preds}

        return {}

    def add_loadings(self, algorithm: AbstractPLS, sub_tag: str = ''):
        """
        Add loadings to the regression group.
        """
        # Add loadings
        if algorithm.has_loadings():
            return {self.merge_if_not_empty(sub_tag, Tags.LOADINGS): algorithm.get_loadings()}

        return {}

    def add_matrices(self, algorithm: AbstractPLS, sub_tag: str = ''):
        """
        Add model matrices to the regression group.
        """
        # Add matrices
        result = {}
        for matrix_name in algorithm.get_matrix_names():
            tag: str = Tags.MATRIX + '-' + matrix_name
            result.update({self.merge_if_not_empty(sub_tag, tag): algorithm.get_matrix(matrix_name)})

        return result

    @classmethod
    def get_datasets(cls) -> Tuple[TestDataset, TestDataset]:
        return TestDataset.BOLTS, TestDataset.BOLTS_RESPONSE
