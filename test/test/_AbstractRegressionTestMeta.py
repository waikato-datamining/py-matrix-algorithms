#  _AbstractRegressionTestMeta.py
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
from typing import GenericMeta, Tuple, List, Dict, Any
from unittest import TestCase, defaultTestLoader

from .misc import TEST_TAG
from wai.ma.meta import has_tag, will_be_abstract


class AbstractRegressionTestMeta(GenericMeta):
    """
    Meta-class which adds the TestCase class as a base of any classes
    inheriting from AbstractRegressionTest. This is because making
    AbstractRegressionTest inherit from TestCase in the normal fashion
    causes unittest to try to instantiate it, which fails because it
    is abstract. An alternative would be to override the
    __subclasscheck__ method of the meta-class, but for some reason
    it wasn't being called (possible short-cutting in python implementation).
    """
    def __new__(self, name, bases, namespace, **kwargs):
        bases = self.augment_bases(bases, namespace)
        self.ensure_tests(namespace)
        return super().__new__(self, name, bases, namespace, **kwargs)

    @staticmethod
    def augment_bases(bases: Tuple, namespace: Dict) -> Tuple:
        """
        Augments the base classes for the new class with the TestCase class.

        :param bases:       The proposed bases for the class.
        :param namespace:   The namespace of the class.
        :return:            The base classes augmented with the TestCase class.
        """
        if not will_be_abstract(bases, namespace):
            if TestCase not in bases:
                return (*bases, TestCase)
            else:
                return bases
        else:
            if TestCase in bases:
                return tuple(base for base in bases if base is not TestCase)
            else:
                return bases

    @staticmethod
    def ensure_tests(namespace: Dict[str, Any]):
        """
        Makes sure that all methods marked with the test tag will be picked up by unittest.

        :param namespace:   The namespace of the new class.
        :return:            The namespace augmented with the unit-test methods.
        """
        prefix = defaultTestLoader.testMethodPrefix
        new_items = {}  # Required so namespace doesn't change during iteration
        for name, method in namespace.items():
            if callable(method) and has_tag(method, TEST_TAG):
                if not name.startswith(prefix):
                    test_name = prefix + '_' + name
                    if test_name in namespace:
                        continue  # Test method already present
                    new_items[test_name] = method
        namespace.update(new_items)
