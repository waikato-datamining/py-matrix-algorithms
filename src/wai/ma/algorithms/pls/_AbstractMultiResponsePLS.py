#  _AbstractMultiResponsePLS.py
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
from abc import abstractmethod

from ...core.errors import MatrixAlgorithmsError
from ...core.matrix import Matrix
from ._AbstractPLS import AbstractPLS


class AbstractMultiResponsePLS(AbstractPLS):
    """
    Ancestor for PLS algorithms that work on multiple response variables.
    """
    @abstractmethod
    def get_min_columns_response(self) -> int:
        """
        Returns the minimum number of columns ther response matrix has to have.

        :return:    The minimum.
        """
        pass

    @abstractmethod
    def get_max_columns_response(self) -> int:
        """
        Returns the maximum number of columns the response matrix has to have.

        :return:    The maximum, -1 for unlimited.
        """
        pass

    def _do_configure(self, X: Matrix, y: Matrix):
        if y.num_columns() < self.get_min_columns_response():
            raise MatrixAlgorithmsError(f"Algorithm requires at least {self.get_min_columns_response()} response "
                                        f"columns, found: {y.num_columns()}")
        elif self.get_max_columns_response() != -1 and y.num_columns() > self.get_max_columns_response():
            raise MatrixAlgorithmsError(f"Algorithm can handle at most {self.get_max_columns_response()} response "
                                        f"columns, found: {y.num_columns()}")

        super()._do_configure(X, y)
