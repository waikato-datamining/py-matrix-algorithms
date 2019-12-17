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
from typing import Tuple

from test.test import TestDataset
from wai.ma.algorithm.glsw import YGradientGLSW
from wai.ma.core.matrix import Matrix

from ._GLSWTest import GLSWTest


class YGradientGLSWTest(GLSWTest):
    @classmethod
    def subject_type(cls):
        return YGradientGLSW

    def standard_regression(self, subject: YGradientGLSW, *resources: Matrix):
        # Get inputs: Simulate second instrument as x1 with noise
        X, y = resources

        # Init GLSW
        subject.initialize(X, y)

        # Add regressions
        return self.add_glsw_regressions(subject, X)

    @classmethod
    def get_datasets(cls) -> Tuple[TestDataset, TestDataset]:
        return TestDataset.BOLTS, TestDataset.BOLTS_RESPONSE
