#  _MultiFilter.py
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

from ..core.algorithm import MatrixAlgorithm, UnsupervisedMatrixAlgorithm
from ..core.matrix import Matrix


class MultiFilter(MatrixAlgorithm):
    """
    Filter which encapsulates a series of sub-filters, and applies
    them in a given order.
    """
    def __init__(self, algorithms: List[MatrixAlgorithm]):
        super().__init__()

        # The algorithms in the order of application
        self._algorithms: List[MatrixAlgorithm] = algorithms
        
    def _do_transform(self, predictors: Matrix) -> Matrix:
        # The result starts as the predictors
        result: Matrix = predictors
        
        # Apply each filter in ordered turn
        for algorithm in self._algorithms:
            if isinstance(algorithm, UnsupervisedMatrixAlgorithm):
                result = algorithm.configure_and_transform(result)
            else:
                result = algorithm.transform(result)
            
        return result

    def _do_inverse_transform(self, X: Matrix) -> Matrix:
        # TODO
        return super()._do_inverse_transform(X)

    def is_non_invertible(self) -> bool:
        # We're non-invertible if any sub-algorithm is non-invertible
        return any(map(lambda algorithm: algorithm.is_non_invertible(), self._algorithms))
