from typing import List

from core.error import MatrixAlgorithmsError
from core.matrix import Matrix, real, factory
from transformation.AbstractTransformation import AbstractTransformation


class SavitzkyGolayFilter(AbstractTransformation):
    def __init__(self):
        super().__init__()
        self.coef: List[real] = [real(2.0 / 10.0),
                                 real(1.0 / 10.0),
                                 real(0.0),
                                 real(-1.0 / 10.0),
                                 real(-2.0 / 10.0)]

    def configure(self, data: Matrix):
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        mat_extended = self.extend_matrix(data)
        result = factory.zeros_like(mat_extended)

        for i in range(2, mat_extended.num_rows() - 2):
            row_smoothed = self.smooth_row(i, mat_extended)
            result.set_row(i, row_smoothed)

        orig_sized_matrix = self.shrink_matrix(result)
        return orig_sized_matrix

    def shrink_matrix(self, result: Matrix) -> Matrix:
        return result.get_sub_matrix((2, result.num_rows() - 2), (0, result.num_columns()))

    def extend_matrix(self, data: Matrix) -> Matrix:
        first_row: Matrix = data.get_row(0)
        last_row: Matrix = data.get_row(data.num_rows() - 1)
        return first_row\
            .concat_along_rows(first_row)\
            .concat_along_rows(data)\
            .concat_along_rows(last_row)\
            .concat_along_rows(last_row)

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        raise MatrixAlgorithmsError('Inverse transformation of Savitzky Golay is not available.')

    def smooth_row(self, i: int, matrix: Matrix) -> Matrix:
        res: Matrix = factory.zeros(1, matrix.num_columns())

        window_size: int = len(self.coef)
        for m in range(window_size):
            coef_idx = window_size - 1 - m
            row_idx = i - (m - 2)
            row = matrix.get_row(row_idx)
            row_scaled = row.mul(self.coef[coef_idx])
            res = res.add(row_scaled)

        return res
