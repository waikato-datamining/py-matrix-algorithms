#  _AbstractSingleResponsePLS.py
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
from abc import ABC
from typing import Optional

from ._AbstractPLS import AbstractPLS
from ...core import real, NAN, PreprocessingType, ONE, ZERO
from ...core.matrix import Matrix, helper
from ...transformation import AbstractTransformation, Center, Standardize


class AbstractSingleResponsePLS(AbstractPLS, ABC):
    def __init__(self):
        super().__init__()
        self.class_mean: real = NAN  # The class mean
        self.class_std_dev: real = NAN  # The class stddev
        self.trans_predictors: Optional[AbstractTransformation] = None  # The transformation for the predictors
        self.trans_response: Optional[AbstractTransformation] = None  # The transformation for the response

    def reset(self):
        """
        Resets the member variables.
        """
        super().reset()

        self.class_mean = NAN
        self.class_std_dev = NAN
        self.trans_predictors = None
        self.trans_response = None

    def check(self, predictors: Optional[Matrix], response: Optional[Matrix]) -> Optional[str]:
        """
        Hook method for checking the data before training.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        result = super().check(predictors, response)

        if result is None:
            if response.num_columns() != 1:
                result = 'Algorithm requires exactly one response variable, found: ' + str(response.num_columns())

        return result

    def do_initialize(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        if self.preprocessing_type is PreprocessingType.CENTER:
            self.class_mean = helper.mean(response, 0)
            self.class_std_dev = ONE
            self.trans_predictors = Center()
            self.trans_response = Center()
        elif self.preprocessing_type is PreprocessingType.STANDARDIZE:
            self.class_mean = helper.mean(response, 0)
            self.class_std_dev = helper.stdev(response, 0)
            self.trans_predictors = Standardize()
            self.trans_response = Standardize()
        elif self.preprocessing_type is PreprocessingType.NONE:
            self.class_mean = ZERO
            self.class_std_dev = ONE
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
                result.set(i, 0, result.get(i, 0) * self.class_std_dev + self.class_mean)

        return result
