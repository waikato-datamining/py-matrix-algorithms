#  _UnconfiguredAlgorithmException.py
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
from typing import Type

from ._MatrixAlgorithmsError import MatrixAlgorithmsError


class UnconfiguredAlgorithmError(MatrixAlgorithmsError):
    """
    Exception thrown when an algorithm that requires configuration
    tries to perform an operation that requires configuration before
    that configuration has occurred.
    """
    def __init__(self, algorithm: Type['MatrixAlgorithm']):
        super().__init__(f"Algorithm {algorithm.__name__} requires configuration")
