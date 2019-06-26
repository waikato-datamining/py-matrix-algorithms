#  _AbstractPLS.py
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
from abc import abstractmethod
from typing import List, Optional

from ...core import SupervisedFilter
from ...algorithm import AbstractAlgorithm
from ...core import PreprocessingType
from ...core.matrix import Matrix


class AbstractPLS(AbstractAlgorithm, SupervisedFilter):
    def __init__(self):
        super().__init__()
        self.preprocessing_type: PreprocessingType = PreprocessingType.NONE  # The preprocessing type to perform
        self.num_components: int = 5  # The maximum number of components to generate

    def initialize(self, predictors: Optional[Matrix] = None, response: Optional[Matrix] = None) -> Optional[str]:
        if predictors is None and response is None:
            """
            Initializes the members.
            """
            super().initialize()
            self.num_components = 5
            self.preprocessing_type = PreprocessingType.NONE
        else:
            """
            Initialises using the provided data.

            :param predictors:  The input data.
            :param response:    The dependent variable(s).
            :return:            None if successful, otherwise error message.
            """
            # Always work on copies
            predictors = predictors.copy()
            response = response.copy()

            self.reset()

            result = self.check(predictors, response)

            if result is None:
                result = self.do_initialize(predictors, response)
                self.initialised = result is None

            return result

    @staticmethod
    def validate_preprocessing_type(value: PreprocessingType) -> bool:
        return True

    @staticmethod
    def validate_num_components(value: int) -> bool:
        return True

    @abstractmethod
    def get_matrix_names(self) -> List[str]:
        """
        Returns all the available matrices.

        :return:    The names of the matrices.
        """
        pass

    @abstractmethod
    def get_matrix(self, name: str) -> Optional[Matrix]:
        """
        Returns the matrix with the specified name.

        :param name:    The name of the matrix.
        :return:        The matrix, None if not available.
        """
        pass

    @abstractmethod
    def has_loadings(self) -> bool:
        """
        Whether the algorithm supports the return of loadings.

        :return:    True if supported.
        """
        pass

    @abstractmethod
    def get_loadings(self) -> Optional[Matrix]:
        """
        Returns the loadings.

        :return:    The loadings, None if not available.
        """
        pass

    def check(self, predictors: Optional[Matrix], response: Optional[Matrix]) -> Optional[str]:
        """
        Hook method for checking the data before training.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        if predictors is None:
            return 'No predictors matrix provided!'
        if response is None:
            return 'No response matrix provided!'
        return None

    @abstractmethod
    def do_initialize(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        """
        Trains using the provided data.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        pass

    @abstractmethod
    def can_predict(self) -> bool:
        """
        Returns whether the algorithm can make predictions.

        :return:    True if can make predictions.
        """
        pass

    @abstractmethod
    def do_predict(self, predictors: Matrix) -> Matrix:
        """
        Performs predictions on the data.

        :param predictors:  The input data.
        :return:            The predictions.
        """
        pass

    def predict(self, predictors: Matrix) -> Matrix:
        """
        Performs predictions on the data.

        :param predictors:  The input data.
        :return:            The predictions.
        """
        if not self.is_initialised():
            raise RuntimeError('Algorithm has not been initialised!')

        return self.do_predict(predictors)

    @abstractmethod
    def do_transform(self, predictors: Matrix) -> Matrix:
        """
        Transforms the data.

        :param predictors:  The input data.
        :return:            The transformed data.
        """
        pass

    def transform(self, predictors: Matrix) -> Matrix:
        """
        Transforms the data.

        :param predictors:  The input data.
        :return:            The transformed data.
        """
        if not self.is_initialised():
            raise RuntimeError('Algorithm has not been initialised!')

        return self.do_transform(predictors)

    @abstractmethod
    def do_perform_initialization(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        """
        Initialises using the provided data.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        pass

    @abstractmethod
    def do_perform_predictions(self, predictors: Matrix) -> Matrix:
        """
        Performs predictions on the data.

        :param predictors:  The input data.
        :return:            The transformed data and the predictions.
        """
        pass

    def to_string(self) -> str:
        """
        For outputting some information about the algorithm.

        :return:    The information.
        """
        result = self.__class__.__name__ + '\n'
        result += self.__class__.__name__.replace('.', '=') + '\n\n'
        result += 'Debug        : ' + str(self.debug) + '\n'
        result += '# components : ' + str(self.num_components) + '\n'
        result += 'Preprocessing: ' + str(self.preprocessing_type) + '\n'

        return result
