#  _AbstractTransformation.py
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

from abc import abstractmethod, ABC

from ..core import Filter
from ..core.matrix import Matrix


class AbstractTransformation(Filter, ABC):
    def __init__(self):
        self.configured: bool = False

    def reset(self):
        self.configured = False

    @abstractmethod
    def configure(self, data: Matrix):
        pass

    @abstractmethod
    def do_transform(self, data: Matrix) -> Matrix:
        pass

    def transform(self, data: Matrix) -> Matrix:
        if not self.configured:
            self.configure(data)
        return self.do_transform(data)

    @abstractmethod
    def do_inverse_transform(self, data: Matrix) -> Matrix:
        pass

    def inverse_transform(self, data: Matrix) -> Matrix:
        if not self.configured:
            self.configure(data)
        return self.do_inverse_transform(data)

    @classmethod
    def quick_apply(cls, data: Matrix) -> Matrix:
        return cls().transform(data)

