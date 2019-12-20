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

from wai.common import switch, case, default, break_

from ...core.algorithm import PredictingSupervisedMatrixAlgorithm, UnsupervisedMatrixAlgorithm
from ...core.matrix import Matrix
from .._Center import Center
from .._Standardize import Standardize
from ._PreprocessingType import PreprocessingType


class AbstractPLS(PredictingSupervisedMatrixAlgorithm):
    def __init__(self):
        super().__init__()

        self._preprocessing_type: PreprocessingType = PreprocessingType.NONE  # The preprocessing type to perform
        self._num_components: int = 5  # The maximum number of components to generate

        self._trans_predictors: Optional[UnsupervisedMatrixAlgorithm] = None
        self._trans_response: Optional[UnsupervisedMatrixAlgorithm] = None

    def get_preprocessing_type(self) -> PreprocessingType:
        return self._preprocessing_type

    def set_preprocessing_type(self, value: PreprocessingType):
        self._preprocessing_type = value
        self.reset()

    preprocessing_type = property(get_preprocessing_type, set_preprocessing_type)

    def get_num_components(self) -> int:
        return self._num_components

    def set_num_components(self, value: int):
        self._num_components = value
        self.reset()

    num_components = property(get_num_components, set_num_components)

    def _do_reset(self):
        super()._do_reset()

        self._trans_predictors = None
        self._trans_response = None

    def _do_configure(self, X: Matrix, y: Matrix):
        with switch(self._preprocessing_type):
            if case(PreprocessingType.CENTER):
                self._trans_predictors = Center()
                self._trans_response = Center()
                break_()
            if case(PreprocessingType.STANDARDIZE):
                self._trans_predictors = Standardize()
                self._trans_response = Standardize()
                break_()
            if case(PreprocessingType.NONE):
                self._trans_predictors = None
                self._trans_response = None
                break_()
            if default():
                raise RuntimeError(f"Unhandled preprocessing type: {self._preprocessing_type}")

        if self._trans_predictors is not None:
            X = self._trans_predictors.configure_and_transform(X)
        if self._trans_response is not None:
            y = self._trans_response.configure_and_transform(y)

        self._do_pls_configure(X, y)

    @abstractmethod
    def _do_pls_configure(self, X: Matrix, y: Matrix):
        """
        PLS-specific configuration implementation. Override to configure
        the PLS algorithm on the given matrices, after feature/target
        normalisation has been performed.

        :param X:   The normalised feature configuration matrix.
        :param y:   The normalised target configuration matrix.
        """
        pass

    def _do_transform(self, X: Matrix) -> Matrix:
        if self._trans_predictors is not None:
            X = self._trans_predictors.transform(X)

        return self._do_pls_transform(X)

    @abstractmethod
    def _do_pls_transform(self, X: Matrix) -> Matrix:
        """
        Internal implementation of PLS transformation. Override
        to implement the PLS-specific transformation code, after
        normalisation has been performed.

        :param X:   The normalised matrix to apply the algorithm to.
        :return:    The normalised matrix resulting from the transformation.
        """
        pass

    def _do_predict(self, X: Matrix) -> Matrix:
        if self._trans_predictors is not None:
            X = self._trans_predictors.transform(X)

        result: Matrix = self._do_pls_predict(X)

        if self._trans_response is not None:
            result = self._trans_response.inverse_transform(result)

        return result

    def _do_pls_predict(self, X: Matrix) -> Matrix:
        """
        PLS-specific prediction implementation. Override to predict
        normalised target values for the given normalised feature matrix.

        :param X:   The normalised feature matrix to predict against.
        :return:    The normalised predictions.
        """
        pass

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

    @abstractmethod
    def can_predict(self) -> bool:
        """
        Returns whether the algorithm can make predictions.

        :return:    True if can make predictions.
        """
        pass

    def is_non_invertible(self) -> bool:
        return True

    def to_string(self) -> str:
        """
        For outputting some information about the algorithm.

        :return:    The information.
        """
        result = self.__class__.__name__ + '\n'
        result += self.__class__.__name__.replace('.', '=') + '\n\n'
        result += 'Debug        : ' + str(self.debug) + '\n'
        result += '# components : ' + str(self._num_components) + '\n'
        result += 'Preprocessing: ' + str(self._preprocessing_type) + '\n'

        return result
