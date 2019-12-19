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

from ..core import real, utils
from ..core.algorithm import MatrixAlgorithm
from ..core.matrix import Matrix, factory
from ._Center import Center
from ._Standardize import Standardize


class PCA(MatrixAlgorithm):
    def __init__(self):
        super().__init__()
        self._variance: real = real(0.95)
        self._max_columns: int = -1
        self._center: bool = False
        self._loadings: Optional[Matrix] = None
        self._scores: Optional[Matrix] = None
        self._correlation: Optional[Matrix] = None
        self._eigenvectors: Optional[Matrix] = None
        self._eigenvalues: Optional[Matrix] = None
        self._sorted_eigens: List[int] = []
        self._sum_of_eigenvalues: real = real(0)
        self._num_cols: int = 0
        self._num_rows: int = 0
        self._keep_cols: List[int] = []
        self._train: Optional[Matrix] = None
        self._transformation: Optional[MatrixAlgorithm] = None

    def get_variance(self) -> real:
        return self._variance

    def set_variance(self, value: real):
        if not (0 < value < 1):
            raise ValueError(f"variance must be in (0, 1), got {value}")

        self._variance = value

    variance = property(get_variance, set_variance)

    def get_max_columns(self) -> int:
        return self._max_columns

    def set_max_columns(self, value: int):
        if value != -1 and value <= 0:
            raise ValueError(f"max_columns must be either -1 or a positive value, got {value}")

        self._max_columns = value

    max_columns = property(get_max_columns, set_max_columns)

    def get_center(self) -> bool:
        return self._center

    def set_center(self, value: bool):
        self._center = value

    center = property(get_center, set_center)

    def get_loadings(self) -> Matrix:
        return self._loadings

    loadings = property(get_loadings)

    def get_scores(self) -> Matrix:
        return self._scores

    scores = property(get_scores)

    def fill_correlation(self):
        """
        Fills the covariance matrix.
        """
        self._correlation = factory.zeros(self._num_cols, self._num_cols)

        for i in range(self._num_cols):
            for j in range(i, self._num_cols):
                cov = real(0)
                for n in range(self._train.num_rows()):
                    cov += self._train.get(n, i) * self._train.get(n, j)

                cov /= self._train.num_rows() - 1
                self._correlation.set(i, j, cov)
                self._correlation.set(j, i, cov)

    def remove_columns(self, data: Matrix) -> Matrix:
        """
        Removes the columns according to self.keep_cols.

        :param data:    The data to trim.
        :return:        The trimmed data.
        """
        if len(self._keep_cols) != data.num_columns():
            rows = []
            for j in range(data.num_rows()):
                rows.append(j)
            data = data.get_sub_matrix(rows, self._keep_cols)

        return data

    def _do_configure(self, instances: Matrix):
        """
        Initialises the filter with the given input data.

        :param instances:   The data to process.
        """
        self._train = instances.copy()

        # Delete any attributes with only one distinct value or are all missing
        self._keep_cols = []
        for j in range(self._train.num_columns()):
            distinct = set()
            for i in range(self._train.num_rows()):
                distinct.add(self._train.get(i, j))
                if len(distinct) > 1:
                    break
            if len(distinct) > 1:
                self._keep_cols.append(j)
        self._train = self.remove_columns(self._train)

        # Transform data
        if self._center:
            self._transformation = Center()
        else:
            self._transformation = Standardize()
        self._train = self._transformation.configure_and_transform(self._train)

        self._num_rows = self._train.num_rows()
        self._num_cols = self._train.num_columns()

        self.fill_correlation()

        # Get eigenvectors/values
        corr = self._correlation.copy()
        self._eigenvectors = corr.get_eigenvectors().copy()
        self._eigenvalues = corr.get_eigenvalues().copy()

        # Any eigenvalues less than 0 are not worth anything -- change to 0
        self._eigenvalues = self._eigenvalues.apply_elementwise(lambda v: real(0) if v < 0 else v)

        self._sorted_eigens = utils.sort(self._eigenvalues.as_native_flattened())
        self._sum_of_eigenvalues = self._eigenvalues.total().as_scalar()

        self._train = None

    def _do_transform(self, data: Matrix) -> Matrix:
        """
        Transform a matrix.

        :param data:    The original data to transform.
        :return:        The transformed data.
        """
        self._do_configure(data)

        num_cols = self._max_columns if self._max_columns > 0 else self._num_cols
        if self._max_columns > 0:
            num_cols_lower_bound = self._num_cols - self._max_columns
        else:
            num_cols_lower_bound = 0
        if num_cols_lower_bound < 0:
            num_cols_lower_bound = 0

        data = self.remove_columns(data)
        data = self._transformation.transform(data)
        values = [[]] * data.num_rows()
        num_cols_act = 0
        for n in range(data.num_rows()):
            new_vals = [real(0)] * num_cols

            cumulative = 0
            cols = 0
            for i in reversed(range(num_cols_lower_bound, self._num_cols)):
                cols += 1
                val = real(0)
                for j in range(self._num_cols):
                    val += self._eigenvectors.get_row(j).get(0, self._sorted_eigens[i]) * data.get(n, j)

                new_vals[self._num_cols - i - 1] = val
                cumulative += self._eigenvalues.get_flat(self._sorted_eigens[i])
                if (cumulative / self._sum_of_eigenvalues) >= self._variance:
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

        self._scores = result
        self._loadings = self.extract_loadings()

        return result

    def is_non_invertible(self) -> bool:
        return True

    def get_coefficients(self) -> Optional[List[List[real]]]:
        if self._eigenvalues is None:
            return None

        if self._max_columns > 0:
            num_cols_lower_bound = self._num_cols - self._max_columns
        else:
            num_cols_lower_bound = 0
        if num_cols_lower_bound < 0:
            num_cols_lower_bound = 0

        # All of the coefficients for a single principle component
        result = []
        cumulative = real(0)

        # Loop through each principle component
        for i in reversed(range(num_cols_lower_bound, self._num_cols)):
            one_pc = []

            for j in range(self._num_cols):
                coeff_value = self._eigenvectors.get(j, self._sorted_eigens[i])
                one_pc.append(coeff_value)

            result.append(one_pc)
            cumulative += self._eigenvalues.get_flat(self._sorted_eigens[i])

            if (cumulative / self._sum_of_eigenvalues) >= self._variance:
                break

        return result

    def extract_loadings(self) -> Matrix:
        """
        Create a matrix to output from the coefficients 2D list.

        :return:    Matrix containing the components.
        """
        coeff = self.get_coefficients()
        result = factory.zeros(self._num_cols, len(coeff) + 1)

        # Add the index column
        for n in range(self._num_cols):
            result.set(n, result.num_columns() - 1, n + 1)

        # Each list is a single column
        for i in range(len(coeff)):
            for n in range(self._num_cols):
                # Column was kept earlier
                value = real(0)
                if n in self._keep_cols:
                    index = self._keep_cols.index(n)
                    if index < len(coeff[i]):
                        value = coeff[i][index]

                result.set(n, i, value)

        return result

    def __str__(self) -> str:
        """
        For outputting some information about the algorithm.

        :return:    The information.
        """
        return (
                   f"{self.__class__.__name__}\n"
                   f"{self.__class__.__name__.replace('.', '=')}\n"
                   f"\n"
                   f"Debug      : {self.debug}\n"
                   f"Variance   : {self._variance}\n"
                   f"Max columns: {self._max_columns}\n"
                   f"Center     : {self._center}\n"
        )




