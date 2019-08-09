#  PassThrough.py
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
from ._AbstractTransformation import AbstractTransformation
from ..core.matrix import Matrix


class PassThrough(AbstractTransformation):
    def configure(self, data: Matrix):
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        return data

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        return data
