#  _PLS1Test.py
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
from typing import TypeVar

from ._AbstractPLSTest import AbstractPLSTest
from wai.ma.algorithm.pls import PLS1

T = TypeVar('T', bound=PLS1)


class PLS1Test(AbstractPLSTest[T]):
    """
    Testcase for the PLS1 algorithm.
    """
    def instantiate_subject(self) -> PLS1:
        return PLS1()
