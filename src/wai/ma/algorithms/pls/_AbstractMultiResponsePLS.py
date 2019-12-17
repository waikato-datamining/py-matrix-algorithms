#  _AbstractMultiResponsePLS.py
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

from ...core import real, PreprocessingType
from ...core.matrix import Matrix, helper
from ...transformation import AbstractTransformation, Center, Standardize
from ._AbstractPLS import AbstractPLS


class AbstractMultiResponsePLS(AbstractPLS):
    """
    Ancestor for PLS algorithms that work on multiple response variables.
    """
    def __init__(self):
        super().__init__()
        self.class_mean: Optional[List[real]] = None
        self.class_std_dev: Optional[List[real]] = None
        self.trans_predictors: Optional[AbstractTransformation] = None
        self.trans_response: Optional[AbstractTransformation] = None

    def reset(self):
        """
        Resets the member variables.
        """
        super().reset()

        self.class_mean = None
        self.class_std_dev = None
        self.trans_predictors = None
        self.trans_response = None

    @abstractmethod
    def get_min_columns_response(self) -> int:
        """
        Returns the minimum number of columns ther response matrix has to have.

        :return:    The minimum.
        """
        pass

    @abstractmethod
    def get_max_columns_response(self) -> int:
        """
        Returns the maximum number of columns the response matrix has to have.

        :return:    The maximum, -1 for unlimited.
        """
        pass

    def check(self, predictors: Optional[Matrix], response: Optional[Matrix]) -> Optional[str]:
        """
        Hook method for checking the data before training.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        result = super().check(predictors, response)

        if result is None:
            if response.num_columns() < self.get_min_columns_response():
                result = 'Algorithm requires at least ' + \
                         str(self.get_min_columns_response()) + \
                         ' response columns, found: ' + \
                         str(response.num_columns())
            elif self.get_max_columns_response() != -1 and response.num_columns() > self.get_max_columns_response():
                result = 'Algorithm can handle at most ' + \
                         str(self.get_max_columns_response()) + \
                         ' response columns, found: ' + \
                         str(response.num_columns())

        return result

    def do_initialize(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        """
        Initializes using the provided data.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        self.class_mean = [real(0)] * response.num_columns()
        self.class_std_dev = [real(0)] * response.num_columns()
        for i in range(response.num_columns()):
            if self.preprocessing_type is PreprocessingType.CENTER:
                self.class_mean[i] = helper.mean(response, 0)
                self.class_std_dev[i] = real(1)
                self.trans_predictors = Center()
                self.trans_response = Center()
            elif self.preprocessing_type is PreprocessingType.STANDARDIZE:
                self.class_mean[i] = helper.mean(response, 0)
                self.class_std_dev[i] = helper.stdev(response, 0)
                self.trans_predictors = Standardize()
                self.trans_response = Standardize()
            elif self.preprocessing_type is PreprocessingType.NONE:
                self.class_mean[i] = real(0)
                self.class_std_dev[i] = real(1)
                self.trans_predictors = None
                self.trans_response = None
            else:
                raise RuntimeError('Unhandled preprocessing type: ' + str(self.preprocessing_type))

        if self.trans_predictors is not None:
            self.trans_predictors.configure(predictors)
            predictors = self.trans_predictors.transform(predictors)
        if self.trans_response is not None:
            self.trans_response.configure(response)
            response = self.trans_response.transform(response)

        result = self.do_perform_initialization(predictors, response)

        return result

    def do_predict(self, predictors: Matrix) -> Matrix:
        """
        Performs predictions on the data.

        :param predictors:  The input data.
        :return:            The transformed data and the predictions.
        """
        result = self.do_perform_predictions(predictors)
        if self.trans_response is not None:
            for i in range(result.num_rows()):
                for j in range(result.num_columns()):
                    result.set(i, j, result.get(i, j) * self.class_std_dev[j] + self.class_mean[j])

        return result
