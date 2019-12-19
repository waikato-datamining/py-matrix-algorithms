#  _GLSWTest.py
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

from wai.test.decorators import RegressionTest

from wai.ma.algorithms.glsw import GLSW
from wai.ma.core.matrix import Matrix, factory

from ...test import Tags
from .._MatrixAlgorithmTest import MatrixAlgorithmTest


class GLSWTest(MatrixAlgorithmTest):
    @classmethod
    def subject_type(cls):
        return GLSW

    def add_glsw_regressions(self, subject: GLSW):
        # Add regressions
        return {
            Tags.PROJECTION: subject.G
        }

    @RegressionTest
    def alpha_1(self, subject: GLSW, *resources: Matrix):
        subject.alpha = 1
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def alpha_100(self, subject: GLSW, *resources: Matrix):
        subject.alpha = 100
        return self.standard_regression(subject, *resources)

    def standard_regression(self, subject: GLSW, *resources: Matrix):
        regressions = super().standard_regression(subject, *resources)
        regressions.update(self.add_glsw_regressions(subject))
        return regressions

    @classmethod
    def common_resources(cls) -> Tuple[Matrix, ...]:
        return cls.glsw_input_data(*super().common_resources())

    @classmethod
    def glsw_input_data(cls, *resources: Matrix) -> Tuple[Matrix, ...]:
        bolts, bolts_response = resources

        # Get inputs: Simulate second instrument as x1 with noise
        x_first_instrument = bolts
        x_second_instrument: Matrix = x_first_instrument.add(factory.randn_like(x_first_instrument, 0, 0.0, 0.1))

        return x_first_instrument, x_second_instrument
