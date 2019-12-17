#  _KernelPLSTest.py
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

from wai.ma.algorithm.pls import KernelPLS
from wai.ma.core.matrix import Matrix
from wai.ma.transformation.kernel import LinearKernel, PolyKernel, RBFKernel

from ._AbstractPLSTest import AbstractPLSTest


class KernelPLSTest(AbstractPLSTest):
    """
    Test case for the KernelPLS algorithm.
    """
    @classmethod
    def subject_type(cls):
        return KernelPLS

    @RegressionTest
    def linear_kernel(self, subject: KernelPLS, *resources: Matrix):
        subject.kernel = LinearKernel()
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def poly_kernel(self, subject: KernelPLS, *resources: Matrix):
        subject.kernel = PolyKernel()
        return self.standard_regression(subject, *resources)

    @RegressionTest
    def rbf_kernel(self, subject: KernelPLS, *resources: Matrix):
        subject.kernel = RBFKernel()
        return self.standard_regression(subject, *resources)
