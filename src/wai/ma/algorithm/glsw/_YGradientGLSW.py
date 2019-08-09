#  _YGradientGLSW.py
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
from typing import List, Optional

from ._GLSW import GLSW
from ...core import real
from ...core.error import MatrixAlgorithmsError
from ...core.matrix import Matrix, factory
from ...transformation import AbstractTransformation


class YGradientGLSW(GLSW):
    def get_covariance_matrix(self, X: Matrix, y: Matrix) -> Matrix:
        y_vals: List[real] = y.to_raw_copy_1D()
        sorted_increasing_row_indices: List[int] = [i for i in range(len(y_vals))]
        sorted_increasing_row_indices.sort(key=lambda o: y_vals[o])

        # Sort increasing
        all_cols: List[int] = [i for i in range(X.num_columns())]
        X_sorted: Matrix = X.get_sub_matrix(sorted_increasing_row_indices, all_cols)
        y_sorted: Matrix = y.get_sub_matrix(sorted_increasing_row_indices, [0])

        sav_golay: SavitzkyGolayFilter = SavitzkyGolayFilter()

        # Apply 5-Point first derivative Savitzky-Golay filter
        X_smoothed: Matrix = sav_golay.transform(X_sorted)
        y_smoothed: Matrix = sav_golay.transform(y_sorted)

        y_smoothed_mean: real = y_smoothed.mean(-1).as_real()
        syd: real = y_smoothed.sub(y_smoothed_mean).pow_elementwise(2).sum(-1).div(y_smoothed.num_rows() - 1).sqrt().as_real()

        # Reweighting matrix
        W: Matrix = factory.zeros(y.num_rows(), y.num_rows())
        for i in range(y_smoothed.num_rows()):
            ydi: real = y_smoothed.get(i, 0)
            W.set(i, i, pow(2.0, -1 * ydi / syd))

        # Covariance matrix
        C: Matrix = X_smoothed.transpose().mul(W.mul(W)).mul(X_smoothed)
        return C

    def check(self, x1: Matrix, x2: Matrix) -> Optional[str]:
        """
        Hook method for checking the data before training.

        :param x1:  First sample set.
        :param x2:  Second sample set.
        :return:    None if successful,
                    otherwise error message.
        """
        if x1 is None:
            return 'No x1 matrix provided!'
        if x2 is None:
            return 'No x2 matrix provided!'
        if x1.num_rows() != x2.num_rows():
            return 'Predictors and response must have the same number of rows!'
        return None


class SavitzkyGolayFilter(AbstractTransformation):
    coef: List[real] = [real(x) for x in [2.0 / 10.0,
                                          1.0 / 10.0,
                                          0.0,
                                          -1.0 / 10.0,
                                          -2.0 / 10.0]]

    def configure(self, data: Matrix):
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        mat_extended: Matrix = self.extend_matrix(data)
        result: Matrix = factory.zeros_like(mat_extended)

        # Smooth all rows
        for i in range(2, mat_extended.num_rows() - 2):
            row_smoothed: Matrix = self.smooth_row(i, mat_extended)
            result.set_row(i, row_smoothed)

        orig_sized_matrix: Matrix = self.shrink_matrix(result)
        return orig_sized_matrix

    def shrink_matrix(self, result: Matrix) -> Matrix:
        """
        Shrink matrix to the original size. Removes first and last 2 rows.

        :param result:  Input matrix.
        :return:        Shrunk matrix.
        """
        return result.get_sub_matrix((2, result.num_rows() - 2), (0, result.num_columns()))

    def extend_matrix(self, data: Matrix) -> Matrix:
        """
        Extend the matrix by 2 rows in the beginning and 2 rows
        at the end (copy first and last elements).

        :param data:    Input matrix.
        :return:        Extended matrix.
        """
        # Extend the matrix by 2 rows at the beginning and 2 rows at the end.
        first_row: Matrix = data.get_row(0)
        last_row: Matrix = data.get_row(data.num_rows() - 1)
        return first_row.concat_along_rows(first_row)\
                        .concat_along_rows(data)\
                        .concat_along_rows(last_row)\
                        .concat_along_rows(last_row)

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        raise MatrixAlgorithmsError('Inverse transform of Savitzky-Golay is not available.')

    def smooth_row(self, i: int, matrix: Matrix) -> Matrix:
        """
        Apply first five-point first gradient Savitzky-Golay smoothing to the i-th
        row of the given matrix.

        :param i:       Row index.
        :param matrix:  Input matrix.
        :return:        Smoothed row.
        """
        res: Matrix = factory.zeros(1, matrix.num_columns())

        window_size: int = len(SavitzkyGolayFilter.coef)
        for m in range(window_size):
            coef_idx: int = (window_size - 1) - m
            row_idx: int = i - (m - 2)
            row: Matrix = matrix.get_row(row_idx)
            row_scaled: Matrix = row.mul(SavitzkyGolayFilter.coef[coef_idx])
            res = res.add(row_scaled)

        return res
