#  _OPLSTest.py
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
from ....algorithm.pls import OPLS, NIPALS, KernelPLS, PLS1, SIMPLS, SparsePLS, PRM


class OPLSTest(AbstractPLSTest[OPLS]):
    def check_transformed_num_components(self):
        # Do nothing since OPLS.transform(X) removes the signal from X_test that is
        # orthogonal to y_train and does not change its shape
        pass

    @TestRegression
    def base_NIPALS(self):
        self.subject.base_PLS = NIPALS()

    @TestRegression
    def base_KernelPLS(self):
        self.subject.base_PLS = KernelPLS()

    @TestRegression
    def base_PLS1(self):
        self.subject.base_PLS = PLS1()

    @TestRegression
    def base_SIMPLS(self):
        self.subject.base_PLS = SIMPLS()

    @TestRegression
    def base_PRM(self):
        self.subject.base_PLS = PRM()

    @TestRegression
    def base_SparsePLS(self):
        self.subject.base_PLS = SparsePLS()

    def instantiate_subject(self) -> OPLS:
        return OPLS()
