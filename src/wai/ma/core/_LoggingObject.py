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
from ..meta import JavaObject
from .java import Logger


class LoggingObject(JavaObject):
    """
    Ancestor of objects with logging support.
    """
    def __init__(self):
        super().__init__()
        self.logger: Logger = Logger.get_logger(self.__class__.__name__)  # The logger to use.
        self.debug: bool = False  # Whether to output debug information.

    def __construct__(self):
        super().__construct__()
        self.initialize()
        self.reset()

    def __setattr__(self, key, value):
        # Validate values
        validator_name = 'validate_' + key
        if hasattr(self, validator_name):
            validator = getattr(self, validator_name)
            if not validator(value):
                raise ValueError('Validation of ' + key + ' failed')

            # If validation passes, reset, unless in __init__ phase
            if self.init_complete():
                self.reset()

        super().__setattr__(key, value)

    def initialize(self):
        """
        For initialising members.

        Default implementation does nothing.
        """
        pass

    def reset(self):
        """
        For resetting data structures when changing parameters.

        Default implementation does nothing.\
        """
        pass
