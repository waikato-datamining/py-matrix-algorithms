#  _NIPALSTest.py
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
from wai.test.decorators import RegressionTest

from wai.ma.algorithm.pls import NIPALS, DeflationMode
from wai.ma.core.matrix import Matrix

from ._AbstractPLSTest import AbstractPLSTest


class NIPALSTest(AbstractPLSTest):
    """
    Test case for the NIPALS algorithm.
    """
    @classmethod
    def subject_type(cls):
        return NIPALS

    @RegressionTest
    def deflation_mode_canonical(self, subject: NIPALS, *resources: Matrix):
        subject.deflation_mode = DeflationMode.CANONICAL
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def deflation_mode_regression(self, subject: NIPALS, *resources: Matrix):
        subject.deflation_mode = DeflationMode.REGRESSION
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def norm_Y_weights_true(self, subject: NIPALS, *resources: Matrix):
        subject.norm_Y_weights = True
        return self.standard_regression(subject, *resources)
