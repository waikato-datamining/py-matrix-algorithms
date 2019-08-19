#  _AbstractMatrixAlgorithmTest.py
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
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Type, Any

from wai.test.decorators import RegressionTest

from wai.ma.core import real
from wai.ma.core.matrix import Matrix
from wai.test import AbstractTest
from wai.test.serialisation import RegressionSerialiser

from ._MatrixSerialiser import MatrixSerialiser
from ._RealSerialiser import RealSerialiser
from ._TestDataset import TestDataset


class AbstractMatrixAlgorithmTest(AbstractTest, ABC):
    @classmethod
    @abstractmethod
    def get_datasets(cls) -> Tuple[TestDataset, ...]:
        """
        Get the input datasets used for the algorithm tests.

        :return:    Paths to input data.
        """
        pass

    @abstractmethod
    def standard_regression(self, subject, *resources) -> Dict[str, Any]:
        """
        Defines the standard way to regression-test the subject with
        the given resources.

        :param subject:     The subject being tested.
        :param resources:   The resources being used to test the subject.
        :return:            The regression map.
        """
        pass

    @classmethod
    def common_resources(cls) -> Tuple[Matrix, ...]:
        return tuple(dataset.load() for dataset in cls.get_datasets())

    @classmethod
    def common_serialisers(cls) -> Dict[Type, Type[RegressionSerialiser]]:
        return {
            Matrix: MatrixSerialiser,
            real: RealSerialiser
        }

    @RegressionTest
    def default_setup(self, subject, *resources):
        """
        Applies the standard regression method to the default subject.

        :param subject:     The default subject.
        :param resources:   Any resources required for the regression test.
        :return:            A regression map.
        """
        return self.standard_regression(subject, *resources)
