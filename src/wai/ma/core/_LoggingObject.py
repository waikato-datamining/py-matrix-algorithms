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
from logging import getLogger, Logger, DEBUG, NOTSET
from typing import Optional


class LoggingObject:
    """
    Ancestor of objects with logging support.
    """
    def get_logger(self) -> Logger:
        if self._logger is None:
            self._logger = getLogger(self.__class__.__qualname__)
        return self._logger

    logger = property(get_logger)

    def get_debug(self):
        return self.logger.isEnabledFor(DEBUG)

    def set_debug(self, value: bool) -> None:
        if value:
            self.logger.setLevel(DEBUG)
        else:
            self.logger.setLevel(NOTSET)

    debug = property(get_debug, set_debug)

    def __init__(self):
        super().__init__()

        self._logger: Optional[Logger] = None
