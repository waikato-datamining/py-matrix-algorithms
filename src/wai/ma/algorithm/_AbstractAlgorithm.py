#  _AbstractAlgorithm.py
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
from abc import abstractmethod

from ..core import LoggingObject


class AbstractAlgorithm(LoggingObject):
    """
    Ancestor for algorithms.
    """
    def __init__(self):
        super().__init__()
        self.initialised: bool = False  # Whether the algorithm has been initialised.

    def is_initialised(self):
        """
        Returns whether the algorithm has been trained.

        :return:    True if trained.
        """
        return self.initialised

    def reset(self):
        """
        Resets the scheme.
        """
        super().reset()
        self.initialised = False

    @abstractmethod
    def to_string(self) -> str:
        pass

    def __str__(self):
        return self.to_string()
