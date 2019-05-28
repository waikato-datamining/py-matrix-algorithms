#  _switch.py
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
from enum import Enum
from typing import Optional


def switch(on) -> 'SwitchContextManager':
    """
    Sugar method for setting up the switch-case context manager.
    Should be called as:

        with switch(on) as case:
            if case([cases]):
                ...
                break_()
            if case([more cases]):
                ...
                break_()
            if case():
                ...

    TODO: Remove need for as clause by inferring the local context
          manager somehow.

    :param on:  The value to be switched on.
    :return:    The switch context manager.
    """
    return SwitchContextManager(on)


class SwitchContextManager:
    """
    Context manager for switch-case blocks.
    """
    def __init__(self, on: Enum):
        self.on = on
        self.matched = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If a break statement was used to exit,
        # don't propagate the exception
        return exc_type is BreakSwitchException

    def __call__(self, *cases: Optional[Enum]):
        # If we match any of the provided cases, always match from
        # now on. No cases provided is the default case.
        if len(cases) == 0 or any(case is self.on for case in cases):
            self.matched = True

        # Allows for fall-through
        return self.matched


def break_():
    """
    Breaks from the switch block.
    """
    raise BreakSwitchException


class BreakSwitchException(Exception):
    pass
