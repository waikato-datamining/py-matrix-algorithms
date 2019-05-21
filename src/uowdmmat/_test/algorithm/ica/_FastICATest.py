#  _FastICATest.py
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

from ...test import AbstractRegressionTest
from ...test.misc import TestRegression, TestDataset
from ....algorithm.ica import FastICA, Algorithm
from ....algorithm.ica.approxfun import LogCosH, Cube, Exponential
from ....core.matrix import Matrix


class FastICATest(AbstractRegressionTest[FastICA]):
    @TestRegression
    def deflation(self):
        self.subject.algorithm = Algorithm.DEFLATION

    @TestRegression
    def parallel(self):
        self.subject.algorithm = Algorithm.PARALLEL

    @TestRegression
    def white_false(self):
        self.subject.whiten = False

    @TestRegression
    def logcosh(self):
        self.subject.fun = LogCosH()

    @TestRegression
    def cube(self):
        self.subject.fun = Cube()

    @TestRegression
    def exp(self):
        self.subject.fun = Exponential()

    def setup_regressions(self, subject: FastICA, input_data: List[Matrix]):
        X: Matrix = input_data[0]

        transform: Matrix = self.subject.transform(X)
        self.add_regression('transform', transform)
        self.add_regression('components', self.subject.components)
        self.add_regression('mixing', self.subject.mixing)
        self.add_regression('sources', self.subject.sources)

    def get_datasets(self) -> List[TestDataset]:
        return [TestDataset.BOLTS]

    def instantiate_subject(self) -> FastICA:
        return FastICA()
