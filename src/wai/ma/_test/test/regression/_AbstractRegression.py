#  _AbstractRegression.py
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
from abc import abstractmethod, ABCMeta
from unittest import TestCase

from ....meta import print_stack_trace
from ....core import real


class AbstractRegression(TestCase, metaclass=ABCMeta):
    """
    Class representing an abstract regression test regarding a single reference file.
    """

    EPSILON: real = real(1e-6)  # Tolerance

    def __init__(self, path: str, actual):
        super().__init__()
        self.path: str = path + '.' + self.get_filename_extension()  # Regression path
        self.actual = actual  # Actual result
        self.expected = None  # Expected result
        self.__load_expected()

    @staticmethod
    def ensure_dir_exists(path: str):
        sub_path: str = os.path.dirname(path)
        os.makedirs(sub_path, exist_ok=True)

    def get_path(self) -> str:
        return self.path

    @abstractmethod
    def check_equals(self, expected, actual):
        """
        Check the actual and expected results for equality.
        """
        pass

    def run_assertions(self):
        """
        Run all assertions.
        """
        self.check_equals(self.expected, self.actual)

    @abstractmethod
    def read_expected(self, path: str):
        """
        Read the expected result from the reference path.

        :param path:    Reference file path.
        :return:        Expected/Reference result.
        """
        pass

    @abstractmethod
    def write_expected(self, path: str, expected):
        """
        Write the expected result in the case when no expected result could be found.

        :param path:        Reference file path.
        :param expected:    Expected object.
        """
        pass

    def __load_expected(self):
        """
        Loads the expected results.
        """
        try:
            self.expected = self.read_expected(self.path)
        except FileNotFoundError:
            print('File <' + self.path + '> does not exist yet. Creating new reference.')
            self.create_new_reference(self.path)
            self.expected = self.actual
        except:
            print_stack_trace()
            self.fail('Failed to load reference file: ' + self.path)

    def create_new_reference(self, path: str):
        """
        Create new reference file.

        :param path:    File path to store the reference at.
        """
        try:
            self.ensure_dir_exists(path)
            self.write_expected(path, self.actual)
        except:
            print_stack_trace()
            self.fail('Failed to create new reference file: ' + path)

    @abstractmethod
    def get_filename_extension(self) -> str:
        """
        Gets the filename extension. E.g. one of [txt, csv, dat, ref, ...]

        :return:    Filename extension
        """
        pass