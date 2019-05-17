#  _RegressionManager.py
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
from typing import List, Union

from .regression import AbstractRegression, MatrixRegression, RealRegression
from ...core import real
from ...core.matrix import Matrix


class RegressionManager:
    """
    Group of regressions. Contains all parameters/outputs/model-matrices that
    need to be checked for the regression tests of a single model configuration.

    Each object that should be checked has to be added via one of the add methods.
    """
    def __init__(self, reference_dir: str, test_name: str):
        self.reference_dir: str = reference_dir
        self.regressions: List[AbstractRegression] = []
        self.test_name: str = test_name

    def add(self, tag: str, value: Union[Matrix, real]):
        path: str = self.construct_path(tag)
        if isinstance(value, Matrix):
            regression: AbstractRegression = MatrixRegression(path, value)
        else:
            regression: AbstractRegression = RealRegression(path, value)
        self.regressions.append(regression)

    def construct_path(self, tag: str) -> str:
        return os.path.join(self.reference_dir, self.test_name, tag)

    def run_assertions(self):
        for reg in self.regressions:
            reg.run_assertions()
