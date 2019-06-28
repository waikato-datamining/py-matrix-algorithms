#  _YGradientGLSWTest.py
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
from typing import List, TypeVar

from ._GLSWTest import GLSWTest
from ...test.misc import TestDataset
from ....algorithm.glsw import GLSW, YGradientGLSW
from ....core.matrix import Matrix

T = TypeVar('T', bound=YGradientGLSW)


class YGradientGLSWTest(GLSWTest[T]):
    def setup_regressions(self, subject: YGradientGLSW, input_data: List[Matrix]):
        # Get inputs: Simulate second instrument as x1 with noise
        X: Matrix = input_data[0]
        y: Matrix = input_data[1]

        # Init GLSW
        subject.initialize(X, y)

        # Add regressions
        self.add_glsw_regressions(subject, X)

    def get_datasets(self) -> List[TestDataset]:
        return [TestDataset.BOLTS, TestDataset.BOLTS_RESPONSE]

    def instantiate_subject(self) -> YGradientGLSW:
        return YGradientGLSW()