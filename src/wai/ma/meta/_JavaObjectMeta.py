#  _JavaObjectMeta.py
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
from abc import ABCMeta


class JavaObjectMeta(ABCMeta):
    """
    Meta-class for creating objects that are more Java-like.
    """
    def __call__(cls, *args, **kwargs):
        # Create the object as normal
        instance = super().__call__(*args, **kwargs)

        # If the class defines a constructor method, call it
        if hasattr(instance, '__construct__'):
            instance.__construct__(*args, **kwargs)

        # Return the new instance
        return instance
