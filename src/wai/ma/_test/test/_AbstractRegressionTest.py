#  _AbstractRegressionTest.py
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
from abc import abstractmethod
from typing import Optional, List, Union, TypeVar, Generic
from unittest import TestCase

from ._AbstractRegressionTestMeta import AbstractRegressionTestMeta
from .misc import TestDataset, TestRegression, REGRESSION_TAG
from ...core import real
from ...meta import print_stack_trace, has_tag
from ._RegressionManager import RegressionManager
from ...core.matrix import Matrix

T = TypeVar('T')


class AbstractRegressionTest(TestCase, Generic[T], metaclass=AbstractRegressionTestMeta):
    """
    Base class of all regression tests. Uses AbstractRegressionTestMeta to ensure
    TestCase is a base of all concrete sub-classes without actually being a base
    of this class. Hence method calls to TestCase methods show as warnings, but
    will be resolved at run-time.
    """
    def __init__(self, *args, **kwargs):
        args = (args[0][5:],)  # Strip the 'test_' prefix from the method name
        super().__init__(*args, **kwargs)
        self.test_name: str = ''  # TestName reference
        self.input_data: Optional[List[Matrix]] = None  # Regression input matrices
        self.regression_manager: Optional[RegressionManager] = None  # Regression manager
        self.subject: T = None  # Subject to test

    @TestRegression
    def default_setup(self):
        pass

    def before_each(self, test_info):
        self.input_data = self.get_input_data()
        self.subject = self.instantiate_subject()
        self.test_name = self._testMethodName.replace('()', '')
        self.regression_manager = RegressionManager(self.get_reference_dir(), self.test_name)

    def after_each(self, test_info):
        # Check if the test contains the regression tag
        method = getattr(self, self.test_name, None)
        if has_tag(method, REGRESSION_TAG):
            self.run_regression()

    @abstractmethod
    def setup_regressions(self, subject: T, input_data: List[Matrix]):
        """
        Setup the given regressions. First run the algorithm on the data. Then
        add all components of the model via add_regression or other add methods.

        :param subject:     Algorithm to run.
        :param input_data:  Input data.
        """
        pass

    @abstractmethod
    def get_datasets(self) -> List[TestDataset]:
        """
        Get the input datasets used for the algorithm tests.

        :return:    Paths to input data.
        """
        pass

    @abstractmethod
    def instantiate_subject(self) -> T:
        """
        Create an instance of the subject.
        :return:
        """
        pass

    def run_regression(self):
        """
        Run the set-up algorithm.
        """
        try:
            self.setup_regressions(self.subject, self.input_data)
        except:
            print_stack_trace()
            self.fail('Setting up regression group failed.')

        self.regression_manager.run_assertions()

    def get_input_data(self):
        """
        Get the input data.

        :return:    Input data.
        """
        return [dataset.load() for dataset in self.get_datasets()]

    def get_reference_dir(self) -> str:
        """
        Get reference file directory.
        Constructed based on the package name and removes "Test" from each class name.

        :return:    Reference file directory.
        """
        # Get the fully-qualified name of the class (in dotted form)
        fully_qualified_name = self.__class__.__module__ + '.' + self.__class__.__qualname__

        # Split into sub-level parts
        name_parts = fully_qualified_name.split('.')

        # Remove any "private" levels (module/package names starting with _)
        name_parts = [part for part in name_parts if not part.startswith('_')]

        # Remove the "Test" suffix from the class name
        if name_parts[-1].endswith('Test'):
            name_parts[-1] = name_parts[-1][:-4]

        return os.path.join('..', 'resources', 'regression', *name_parts)

    def add_regression(self, tag: str, value: Union[Matrix, real]):
        """
        Add a new Matrix/real with a tag to the regression manager.

        :param tag:     Tag for the value.
        :param value:   Value to check with the reference.
        """
        self.regression_manager.add(tag, value)

    def setUp(self) -> None:
        self.before_each(None)

    def tearDown(self) -> None:
        self.after_each(None)
