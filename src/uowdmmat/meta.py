#  meta.py
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

"""
Collection of tools to aid in the programming of this library.
"""

import traceback
from typing import Dict, Tuple, Any


def print_stack_trace():
    """
    Prints the stack-trace to stderr.
    """
    traceback.print_exc()


def Tag(tag: str):
    """
    Decorator that adds the given tag to the decorated function.

    :param tag:     The tag to add to the function.
    """
    def apply_tag(method):
        if not hasattr(method, '__tags'):
            method.__tags = set()
        method.__tags.add(tag)
        return method

    return apply_tag


def has_tag(method, tag: str) -> bool:
    """
    Checks if the given method has the given tag.

    :param method:  The method to check.
    :param tag:     The tag to check for.
    :return:        True if the tag exists on the method,
                    False if not.
    """
    return hasattr(method, '__tags') and tag in method.__tags


def decorator_sequence(*decorators):
    """
    Helper method which creates a decorator that applies the given
    sub-decorators. Decorators are applied in reverse order given.

    @decorator_sequence(dec_1, dec_2, ...)
    def function(...):
        ...

    is equivalent to:

    @dec_1
    @dec_2
    ...
    def function(...):
        ...

    :param decorators:  The sub-decorators.
    :return:            A function which applies the given decorators.
    """
    def apply(obj):
        for dec in reversed(decorators):
            obj = dec(obj)
        return obj

    return apply


def get_abstract_methods(bases: Tuple, namespace: Dict[str, Any]):
    """
    Gets the names of the abstract methods that will result from
    a class created with the given base-class set and namespace.

    :param bases:       The base-class set.
    :param namespace:   The namespace.
    :return:            The set of abstract method names.
    """
    abstract_methods = set()

    for base in bases:
        abstract_methods.update(getattr(base, '__abstractmethods__', set()))

    for name, value in namespace.items():
        if is_abstract_function(value):
            abstract_methods.add(name)
        else:
            if name in abstract_methods:
                abstract_methods.remove(name)

    return abstract_methods


def will_be_abstract(bases: Tuple, namespace: Dict[str, Any]):
    """
    Determines if a class made with the given set of base-classes
    and namespace will be abstract or concrete.

    :param bases:       The set of base classes.
    :param namespace:   The namespace of the new class.
    :return:            True if the newly-created class will be abstract,
                        False if it will be concrete.
    """
    return len(get_abstract_methods(bases, namespace)) > 0


def is_abstract_function(func):
    return getattr(func, '__isabstractmethod__', False)

