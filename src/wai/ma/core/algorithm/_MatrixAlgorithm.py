#  _MatrixAlgorithm.py
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

from ..errors import UninvertibleAlgorithmError
from ..matrix import Matrix
from .._LoggingObject import LoggingObject


class MatrixAlgorithm(LoggingObject, ABC):
    """
    Base class for all matrix algorithms.
    """
    def transform(self, X: Matrix) -> Matrix:
        """
        Performs the transformation that this algorithm represents on
        the given matrix.

        :param X:   The matrix to apply the algorithm to.
        :return:    The matrix resulting from the transformation.
        """
        if X is None:
            raise ValueError("Can't transform null matrix")

        return self._do_transform(X)

    @abstractmethod
    def _do_transform(self, X: Matrix) -> Matrix:
        """
        Internal implementation of algorithm transformation. Override
        to implement the transformation-specific code.

        :param X:   The matrix to apply the algorithm to.
        :return:    The matrix resulting from the transformation.
        """
        pass

    def inverse_transform(self, X: Matrix) -> Matrix:
        """
        Performs the inverse of the transformation that this algorithm
        represents on the given matrix.

        :param X:   The matrix to inverse-apply the algorithm to.
        :return:    The matrix resulting from the inverse-transformation.
        """
        if X is None:
            raise ValueError("Can't inverse-transform null matrix")

        return self._do_inverse_transform(X)

    def _do_inverse_transform(self, X: Matrix) -> Matrix:
        """
        Internal implementation of algorithm inverse-transformation. Override
        to implement the transformation-specific code.

        :param X:   The matrix to inverse-apply the algorithm to.
        :return:    The matrix resulting from the inverse-transformation.
        """
        raise UninvertibleAlgorithmError(self.__class__)

    def is_non_invertible(self) -> bool:
        """
        Whether the algorithm is currently non-invertible. If it's
        not certain whether an inversion will fail (e.g. it depends
        on the input), this method should return false, and calling
        {@link MatrixAlgorithm#inverseTransform(Matrix)} will throw
        {@link InverseTransformException} if it does fail. Meant to
        provide a shortcut where performing the inverse-transform
        may be expensive if it is possible but it is also possible to
        tell in advance in some cases that it is not possible. Also
        used for algorithms that are always impossible to invert.

        :return:    Whether the algorithm is currently definitely impossible
                    to invert.
        """
        return False
