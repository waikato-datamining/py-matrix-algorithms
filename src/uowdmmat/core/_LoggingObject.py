#  _LoggingObject.py
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
from abc import ABC

from .java import Logger


class LoggingObject(ABC):
    """
    Ancestor of objects with logging support.
    """
    def __init__(self):
        self.logger = None  # The logger to use.
        self.debug = False  # Whether to output debug information.
        self.initialize()
        self.reset()

    def initialize(self):
        """
        For initialising members.

        Default implementation does nothing.
        """
        self.get_logger()

    def reset(self):
        """
        For resetting data structures when changing parameters.

        Default implementation does nothing.\
        """
        pass

    def get_logger(self):
        """
        Returns the logger.

        :return:    The logger.
        """
        if self.logger is None:
            self.logger = Logger.get_logger(self.__name__)
        return self.logger
