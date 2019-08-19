#  _RealSerialiser.py
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
from typing import IO, Optional

from wai.ma.core import real
from wai.test.serialisation import RegressionSerialiser

from ._epsilon import EPSILON


class RealSerialiser(RegressionSerialiser[real]):
    @classmethod
    def binary(cls) -> bool:
        return False

    @classmethod
    def extension(cls) -> str:
        return "txt"

    @classmethod
    def serialise(cls, result: real, file: IO[str]):
        file.write(str(result))

    @classmethod
    def deserialise(cls, file: IO[str]) -> real:
        return real(file.read())

    @classmethod
    def compare(cls, result: real, reference: real) -> Optional[str]:
        if abs(result - reference) > EPSILON:
            return "Result: " + str(result) + ", reference: " + str(reference)
        else:
            return None
