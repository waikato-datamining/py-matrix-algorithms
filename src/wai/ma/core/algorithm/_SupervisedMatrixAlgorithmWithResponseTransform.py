#  _SupervisedMatrixAlgorithmWithResponseTransform.py
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
from abc import ABC, abstractmethod

from ..matrix import Matrix
from ._SupervisedMatrixAlgorithm import SupervisedMatrixAlgorithm


class SupervisedMatrixAlgorithmWithResponseTransform(SupervisedMatrixAlgorithm, ABC):
    """
    Base-class for algorithms that can transform target matrices as
    well as feature matrices.
    """
    def configure_and_transform_response(self, X: Matrix, y: Matrix) -> Matrix:
        """
        Transforms the target matrix, configuring the algorithm on the matrices
        if it is not already configured.

        :param X:   The feature matrix to configure on.
        :param y:   The target matrix to configure on and apply the algorithm to.
        :return:    The matrix resulting from the transformation.
        """
        if not self.is_configured():
            self.configure(X, y)

        return self.transform_response(y)

    def transform_response(self, y: Matrix) -> Matrix:
        """
        Performs the transformation that this algorithm represents on
        the given target matrix.

        :param y:   The target matrix to apply the algorithm to.
        :return:    The matrix resulting from the transformation.
        """
        # Ensure the algorithm is configured
        self.ensure_configured()

        return self._do_transform_response(y)

    @abstractmethod
    def _do_transform_response(self, y: Matrix) -> Matrix:
        """
        Internal implementation of algorithm transformation. Override
        to implement the transformation-specific code.

        :param y:   The target matrix to apply the algorithm to.
        :return:    The matrix resulting from the transformation.
        """
        pass
