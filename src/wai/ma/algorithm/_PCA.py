#  _PCA.py
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
from typing import Optional, List

from ..transformation import Center, Standardize, AbstractTransformation
from ..core import real, utils, Filter
from ..core.matrix import Matrix, factory
from ._AbstractAlgorithm import AbstractAlgorithm


class PCA(AbstractAlgorithm, Filter):
    def __init__(self):
        super().__init__()
        self.variance: real = real(0.95)
        self.max_columns: int = -1
        self.center: bool = False
        self.loadings: Optional[Matrix] = None
        self.scores: Optional[Matrix] = None
        self.correlation: Optional[Matrix] = None
        self.eigenvectors: Optional[Matrix] = None
        self.eigenvalues: Optional[Matrix] = None
        self.sorted_eigens: List[int] = []
        self.sum_of_eigenvalues: real = real(0)
        self.num_cols: int = 0
        self.num_rows: int = 0
        self.keep_cols: List[int] = []
        self.train: Optional[Matrix] = None
        self.transformation: Optional[AbstractTransformation] = None

    def initialize(self):
        super().initialize()
        self.variance = real(0.95)
        self.max_columns = -1
        self.center = False

    def reset(self):
        super().reset()
        self.loadings = None
        self.scores = None
        self.correlation = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.sorted_eigens = []
        self.sum_of_eigenvalues = real(0)
        self.transformation = None

    @staticmethod
    def validate_variance(value: real):
        return real(0) < value < real(1)

    @staticmethod
    def validate_max_columns(value: int):
        return value == -1 or value > 0

    @staticmethod
    def validate_center(value: bool):
        return True

    def fill_correlation(self):
        """
        Fills the covariance matrix.
        """
        self.correlation = factory.zeros(self.num_cols, self.num_cols)

        for i in range(self.num_cols):
            for j in range(i, self.num_cols):
                cov = real(0)
                for n in range(self.train.num_rows()):
                    cov += self.train.get(n, i) * self.train.get(n, j)

                cov /= self.train.num_rows() - 1
                self.correlation.set(i, j, cov)
                self.correlation.set(j, i, cov)

    def remove_columns(self, data: Matrix) -> Matrix:
        """
        Removes the columns according to self.keep_cols.

        :param data:    The data to trim.
        :return:        The trimmed data.
        """
        if len(self.keep_cols) != data.num_columns():
            rows = []
            for j in range(data.num_rows()):
                rows.append(j)
            data = data.get_sub_matrix(rows, self.keep_cols)

        return data

    def configure(self, instances: Matrix):
        """
        Initialises the filter with the given input data.

        :param instances:   The data to process.
        """
        self.train = instances.copy()

        # Delete any attributes with only one distinct value or are all missing
        self.keep_cols = []
        for j in range(self.train.num_columns()):
            distinct = set()
            for i in range(self.train.num_rows()):
                distinct.add(self.train.get(i, j))
                if len(distinct) > 1:
                    break
            if len(distinct) > 1:
                self.keep_cols.append(j)
        self.train = self.remove_columns(self.train)

        # Transform data
        if self.center:
            self.transformation = Center()
        else:
            self.transformation = Standardize()
        self.train = self.transformation.transform(self.train)

        self.num_rows = self.train.num_rows()
        self.num_cols = self.train.num_columns()

        self.fill_correlation()

        # Get eigenvectors/values
        corr = self.correlation.copy()
        self.eigenvectors = corr.get_eigenvectors().copy()
        self.eigenvalues = corr.get_eigenvalues().copy()

        # Any eigenvalues less than 0 are not worth anything -- change to 0
        self.eigenvalues = self.eigenvalues.apply_elementwise(lambda v: real(0) if v < 0 else v)

        self.sorted_eigens = utils.sort(self.eigenvalues.to_raw_copy_1D())
        self.sum_of_eigenvalues = sum(self.eigenvalues.to_raw_copy_1D())

        self.train = None

    def do_transform(self, data: Matrix) -> Matrix:
        """
        Transform a matrix.

        :param data:    The original data to transform.
        :return:        The transformed data.
        """
        num_cols = self.max_columns if self.max_columns > 0 else self.num_cols
        if self.max_columns > 0:
            num_cols_lower_bound = self.num_cols - self.max_columns
        else:
            num_cols_lower_bound = 0
        if num_cols_lower_bound < 0:
            num_cols_lower_bound = 0

        data = self.remove_columns(data)
        data = self.transformation.transform(data)
        values = [[]] * data.num_rows()
        num_cols_act = 0
        for n in range(data.num_rows()):
            new_vals = [real(0)] * num_cols

            cumulative = 0
            cols = 0
            for i in reversed(range(num_cols_lower_bound, self.num_cols)):
                cols += 1
                val = real(0)
                for j in range(self.num_cols):
                    val += self.eigenvectors.get_row(j).get(0, self.sorted_eigens[i]) * data.get(n, j)

                new_vals[self.num_cols - i - 1] = val
                cumulative += self.eigenvalues.get_from_vector(self.sorted_eigens[i])
                if (cumulative / self.sum_of_eigenvalues) >= self.variance:
                    break

            num_cols_act = max(num_cols_act, cols)
            values[n] = new_vals

        if self.debug:
            self.logger.info('num_cols_act: ' + str(num_cols_act))

        # Generate matrix based on actual number of retained columns
        result = factory.zeros(data.num_rows(), num_cols_act)
        for n in range(len(values)):
            for i in range(min(len(values[n]), num_cols_act)):
                result.set(n, i, values[n][i])

        return result

    def get_coefficients(self) -> Optional[List[List[real]]]:
        if self.eigenvalues is None:
            return None

        if self.max_columns > 0:
            num_cols_lower_bound = self.num_cols - self.max_columns
        else:
            num_cols_lower_bound = 0
        if num_cols_lower_bound < 0:
            num_cols_lower_bound = 0

        # All of the coefficients for a single principle component
        result = []
        cumulative = real(0)

        # Loop through each principle component
        for i in reversed(range(num_cols_lower_bound, self.num_cols)):
            one_pc = []

            for j in range(self.num_cols):
                coeff_value = self.eigenvectors.get(j, self.sorted_eigens[i])
                one_pc.append(coeff_value)

            result.append(one_pc)
            cumulative += self.eigenvalues.get_from_vector(self.sorted_eigens[i])

            if (cumulative / self.sum_of_eigenvalues) >= self.variance:
                break

        return result

    def extract_loadings(self) -> Matrix:
        """
        Create a matrix to output from the coefficients 2D list.

        :return:    Matrix containing the components.
        """
        coeff = self.get_coefficients()
        result = factory.zeros(self.num_cols, len(coeff) + 1)

        # Add the index column
        for n in range(self.num_cols):
            result.set(n, result.num_columns() - 1, n + 1)

        # Each list is a single column
        for i in range(len(coeff)):
            for n in range(self.num_cols):
                # Column was kept earlier
                value = real(0)
                if n in self.keep_cols:
                    index = self.keep_cols.index(n)
                    if index < len(coeff[i]):
                        value = coeff[i][index]

                result.set(n, i, value)

        return result

    def transform(self, data: Matrix) -> Matrix:
        """
        Transforms the data.

        :param data:    The data to transform.
        :return:        The transformed data.
        """
        self.reset()
        self.configure(data)
        result = self.do_transform(data)
        self.scores = result
        self.loadings = self.extract_loadings()

        return result

    def to_string(self) -> str:
        """
        For outputting some information about the algorithm.

        :return:    The information.
        """
        result =    self.__class__.__name__ + '\n'
        result +=   self.__class__.__name__.replace('.', '=') + '\n\n'
        result +=   'Debug      : ' + str(self.debug) + '\n'
        result +=   'Variance   : ' + str(self.variance) + '\n'
        result +=   'Max columns: ' + str(self.max_columns) + '\n'
        result +=   'Center     : ' + str(self.center) + '\n'

        return result





