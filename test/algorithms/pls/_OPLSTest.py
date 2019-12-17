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
from wai.test.decorators import Test, Skip, RegressionTest

from wai.ma.algorithm.pls import OPLS, NIPALS, KernelPLS, PLS1, SIMPLS, SparsePLS, PRM, CCARegression
from wai.ma.core.matrix import Matrix

from ._AbstractPLSTest import AbstractPLSTest


class OPLSTest(AbstractPLSTest):
    @classmethod
    def subject_type(cls):
        return OPLS

    @Test
    @Skip("OPLS.transform(X) removes the signal from X_test that is"
          "orthogonal to y_train and does not change its shape")
    def check_transformed_num_components(self):
        # Do nothing since OPLS.transform(X) removes the signal from X_test that is
        # orthogonal to y_train and does not change its shape
        super().check_transformed_num_components()

    @RegressionTest
    def base_NIPALS(self, subject: OPLS, *resources: Matrix):
        subject.base_PLS = NIPALS()
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def base_KernelPLS(self, subject: OPLS, *resources: Matrix):
        subject.base_PLS = KernelPLS()
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def base_PLS1(self, subject: OPLS, *resources: Matrix):
        subject.base_PLS = PLS1()
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def base_SIMPLS(self, subject: OPLS, *resources: Matrix):
        subject.base_PLS = SIMPLS()
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def base_CCARegression(self, subject: OPLS, *resources: Matrix):
        subject.base_PLS = CCARegression()
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def base_PRM(self, subject: OPLS, *resources: Matrix):
        subject.base_PLS = PRM()
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def base_SparsePLS(self, subject: OPLS, *resources: Matrix):
        subject.base_PLS = SparsePLS()
        return self.standard_regression(subject, *resources)
