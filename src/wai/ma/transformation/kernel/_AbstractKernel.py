#  _AbstractKernel.py
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
from typing import Optional

from ...core import real
from ...core.matrix import Matrix, factory


class AbstractKernel(ABC):
    """
    Abstract kernel class. Implementations represent kernels that compute a dot product of two given
    vectors in the kernel space (see AbstractKernel.applyVector(Matrix, Matrix)).
    That is: K(x,y) = phi(x)*phi(y)
    """
    @abstractmethod
    def apply_vector(self, x: Matrix, y: Matrix) -> real:
        """
        Compute the dot product of the mapped x and y vectors in the kernel space, that is:
        K(x,y) = phi(x)*phi(y)

        :param x:   First vector.
        :param y:   Second vector.
        :return:    Dot product of the given vectors in the kernel space.
        """
        pass

    def apply_matrix(self, X: Matrix, Y: Optional[Matrix] = None) -> Matrix:
        """
        Create a matrix K that consists of entries K_i,j = K(x_i,y_j) = phi(x_i)*phi(y_j)

        :param X:   First matrix.
        :param Y:   Second matrix.
        :return:    Matrix K with K_i,j = K(x_i,y_j) = phi(x_i)*phi(y_j)
        """
        if Y is None:
            return self.__apply_matrix_single_arg(X)
        else:
            result: Matrix = factory.zeros(X.num_rows(), Y.num_rows())
            for i in range(X.num_rows()):
                for j in range(Y.num_rows()):
                    row_i: Matrix = X.get_row(i)
                    row_j: Matrix = Y.get_row(j)
                    value: real = self.apply_vector(row_i, row_j)
                    result.set(i, j, value)
            return result

    def __apply_matrix_single_arg(self, X: Matrix) -> Matrix:
        """
        Create a matrix K that consists of entries K_i,j = K(x_i,x_j) = phi(x_i)*phi(x_j)

        :param X:   First matrix.
        :return:    Matrix K with K_i,j = K(x_i,x_j) = phi(x_i)*phi(x_j)
        """
        n: int = X.num_rows()
        result: Matrix = factory.zeros(n, n)
        for i in range(n):
            for j in range(i, n):
                row_i: Matrix = X.get_row(i)
                row_j: Matrix = X.get_row(j)
                value: real = self.apply_vector(row_i, row_j)
                result.set(i, j, value)
                result.set(j, i, value)
        return result
