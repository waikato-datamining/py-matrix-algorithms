#  _AbstractSingleResponsePLS.py
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

from ...core.errors import MatrixAlgorithmsError
from ...core.matrix import Matrix
from ._AbstractPLS import AbstractPLS


class AbstractSingleResponsePLS(AbstractPLS, ABC):
    """
    Ancestor for PLS algorithms that work on a single response variable.
    """
    def _do_configure(self, X: Matrix, y: Matrix):
        if y.num_columns() != 1:
            raise MatrixAlgorithmsError(f"Algorithm requires exactly one response variable, found {y.num_columns()}")

        super()._do_configure(X, y)
