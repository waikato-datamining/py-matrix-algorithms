#  _CenterTest.py
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
from wai.test.decorators import Test
from wai.ma.core import real
from wai.ma.core.matrix import Matrix
from wai.ma.algorithms import Center

from ._MatrixAlgorithmTest import MatrixAlgorithmTest


class CenterTest(MatrixAlgorithmTest):
    @classmethod
    def subject_type(cls):
        return Center

    @Test
    def mean_is_zero(self, subject: Center, *resources: Matrix):
        bolts, bolts_response = resources
        transform: Matrix = subject.configure_and_transform(bolts)
        actual: real = transform.mean().as_scalar()
        expected: real = real(0.0)
        self.assertAlmostEqual(actual, expected, delta=1e-7)
