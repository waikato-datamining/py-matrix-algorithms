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
from typing import Tuple

from wai.test.decorators import RegressionTest

from wai.ma.algorithm.ica import FastICA, Algorithm
from wai.ma.algorithm.ica.approxfun import LogCosH, Cube, Exponential
from wai.ma.core.matrix import Matrix

from ...test import AbstractMatrixAlgorithmTest, TestDataset


class FastICATest(AbstractMatrixAlgorithmTest):
    @classmethod
    def subject_type(cls):
        return FastICA

    @RegressionTest
    def deflation(self, subject: FastICA, *resources: Matrix):
        subject.algorithm = Algorithm.DEFLATION
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def parallel(self, subject: FastICA, *resources: Matrix):
        subject.algorithm = Algorithm.PARALLEL
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def white_false(self, subject: FastICA, *resources: Matrix):
        subject.whiten = False
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def logcosh(self, subject: FastICA, *resources: Matrix):
        subject.fun = LogCosH()
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def cube(self, subject: FastICA, *resources: Matrix):
        subject.fun = Cube()
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def exp(self, subject: FastICA, *resources: Matrix):
        subject.fun = Exponential()
        return self.standard_regression(subject, *resources)

    def standard_regression(self, subject: FastICA, *resources: Matrix):
        X, = resources

        transform: Matrix = subject.transform(X)

        return {
            'transform': transform,
            'components': subject.components,
            'mixing': subject.mixing,
            'sources': subject.sources
        }

    @classmethod
    def get_datasets(cls) -> Tuple[TestDataset]:
        return TestDataset.BOLTS,
