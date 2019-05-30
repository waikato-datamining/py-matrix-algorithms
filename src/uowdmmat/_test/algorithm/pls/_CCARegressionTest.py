#  _CCARegressionTest.py
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
from ._AbstractPLSTest import AbstractPLSTest
from ...test.misc import TestRegression
from ....algorithm.pls import CCARegression, DeflationMode


class CCARegressionTest(AbstractPLSTest[CCARegression]):
    """
    Test case for the NIPALS algorithm.
    """
    @TestRegression
    def deflation_mode_canonical(self):
        self.subject.deflation_mode = DeflationMode.CANONICAL

    @TestRegression
    def deflation_mode_regression(self):
        self.subject.deflation_mode = DeflationMode.REGRESSION

    @TestRegression
    def norm_Y_weights_true(self):
        self.subject.norm_Y_weights = True

    def instantiate_subject(self) -> CCARegression:
        return CCARegression()
