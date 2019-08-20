#  _Serialisable.py
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
import os
import struct
from abc import ABC, abstractmethod
from io import BytesIO
from typing import IO, Union

# Whether to use little-endianness for serialisation
# (False means big-endian)
LITTLE_ENDIAN: bool = True


class Serialisable(ABC):
    """
    Mixin class to add serialisation to a class.
    """
    @abstractmethod
    def serialise_state(self, stream: IO[bytes]):
        """
        Serialises the state of the object to binary.

        :param stream:  The stream to serialise to.
        """
        pass

    def save_state(self, filename: str):
        """
        Saves the state of this algorithm to file.

        :param filename:    The name of the file to save to.
        """
        # Make sure the directory for the file exists
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # Save the file
        with open(filename, 'wb') as file:
            self.serialise_state(file)

    def get_serialised_state(self) -> bytes:
        """
        Gets the serialised state of this object.
        """
        stream = BytesIO()

        self.serialise_state(stream)

        return stream.read()

    @staticmethod
    def serialise_to_bytes(value: Union[int, float]) -> bytes:
        """
        Helper method to serialise common types to bytes.

        :param value:   The value to serialise.
        :return:        The bytes representation of the value.
        """
        if isinstance(value, int):
            b = value.to_bytes(4, "little" if LITTLE_ENDIAN else "big", signed=True)
            return b
        elif isinstance(value, float):
            b = struct.pack("<d" if LITTLE_ENDIAN else ">d", value)
            return b
        else:
            raise TypeError("serialise_to_bytes only supports int and float types")
