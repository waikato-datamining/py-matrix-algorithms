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
import threading
from enum import Enum
from typing import Optional, List, Dict

context_stacks: Dict[int, List['SwitchContextManager']] = dict()


def add_to_stack(context_manager: 'SwitchContextManager'):
    # Get the thread we're running on
    current_thread: int = threading.get_ident()

    # Ensure there is a context stack for that thread
    if current_thread not in context_stacks:
        context_stacks[current_thread] = []

    # Put ourselves on the stack
    context_stacks[current_thread].append(context_manager)


def pop_from_stack():
    # Get the thread we're running on
    current_thread: int = threading.get_ident()

    # Pop the top of the stack
    context_stacks[current_thread].pop()


def get_current_context_manager() -> 'SwitchContextManager':
    # Get the thread we're running on
    current_thread: int = threading.get_ident()

    # Peek the top of the stack
    return context_stacks[current_thread][-1]


def switch(on) -> 'SwitchContextManager':
    """
    Sugar method for setting up the switch-case context manager.
    Should be called as:

        with switch(on):
            if case([cases]):
                ...
                break_()
            if case([more cases]):
                ...
                break_()
            if default():
                ...

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
        # Add ourselves as the current context manager for the thread
        add_to_stack(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove ourselves as the current context manager for the thread
        pop_from_stack()

        # If a break statement was used to exit,
        # don't propagate the exception
        return exc_type is BreakSwitchException

    def __call__(self, *cases: Optional[Enum]) -> bool:
        # If we match any of the provided cases, always match from
        # now on. No cases provided is the default case.
        if len(cases) == 0 or any(case is self.on for case in cases):
            self.matched = True

        # Allows for fall-through
        return self.matched


def case(*cases: Optional[Enum]) -> bool:
    # Get the current context
    current_context: SwitchContextManager = get_current_context_manager()

    # Call it to check the cases provided
    return current_context(*cases)


def default():
    return case()


def break_():
    """
    Breaks from the switch block.
    """
    raise BreakSwitchException


class BreakSwitchException(Exception):
    pass
