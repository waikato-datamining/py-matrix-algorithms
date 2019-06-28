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
from typing import TypeVar, List

from ...test import AbstractRegressionTest
from ...test.misc import Tags, TestRegression, TestDataset
from ....algorithm.glsw import GLSW
from ....core.matrix import Matrix, factory

T = TypeVar('T', bound=GLSW)


class GLSWTest(AbstractRegressionTest[T]):
    def add_glsw_regressions(self, subject: GLSW, x: Matrix):
        # Add regressions
        self.add_regression(Tags.TRANSFORM, subject.transform(x))
        self.add_regression(Tags.PROJECTION, subject.G)

    @TestRegression
    def alpha_1(self):
        self.subject.alpha = 1

    @TestRegression
    def alpha_100(self):
        self.subject.alpha = 100

    def setup_regressions(self, subject: GLSW, input_data: List[Matrix]):
        # Get inputs: Simulate second instrument as x1 with noise
        x_first_instrument: Matrix = input_data[0]
        x_second_instrument: Matrix = x_first_instrument.add(factory.randn_like(x_first_instrument, 0, 0.0, 0.1))

        # Init GLSW
        subject.initialize(x_first_instrument, x_second_instrument)

        self.add_glsw_regressions(subject, x_first_instrument)

    def get_datasets(self) -> List[TestDataset]:
        return [TestDataset.BOLTS]

    def instantiate_subject(self) -> GLSW:
        return GLSW()