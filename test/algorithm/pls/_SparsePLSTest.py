#  _SparsePLSTest.py
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
from ...test.misc import TestRegression
from ._AbstractPLSTest import AbstractPLSTest
from wai.ma.algorithm.pls import SparsePLS


class SparsePLSTest(AbstractPLSTest[SparsePLS]):
    @TestRegression
    def lambda_0(self):
        self.subject.lambda_ = 0

    @TestRegression
    def lambda_05(self):
        self.subject.lambda_ = 0.001

    def instantiate_subject(self) -> SparsePLS:
        spls: SparsePLS = SparsePLS()
        spls.num_components = 2
        return spls
