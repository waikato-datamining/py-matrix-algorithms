from typing import Optional, List

import numpy as np

from core.matrix import real, Matrix, factory, helper
from transformation.AbstractTransformation import AbstractTransformation


class SavitzkyGolay(AbstractTransformation):
    def __init__(self):
        super().__init__()
        self.polynomial_order: int = 2
        self.derivative_order: int = 1
        self.num_points_left: int = 3
        self.num_points_right: int = 3
        self.coefficients: Optional[Matrix] = None

    def __setattr__(self, key, value):
        # Validate values
        validator_name = 'validate_' + key
        if hasattr(self, validator_name):
            validator = getattr(self, validator_name)
            validator(value)

            # If validation passes, reset
            self.reset()

        super().__setattr__(key, value)

    @staticmethod
    def validate_polynomial_order(value: int):
        if value < 2:
            raise ValueError('polynomial_order must be at least 2')

    @staticmethod
    def validate_derivative_order(value: int):
        if value < 0:
            raise ValueError('derivative_order must be at least 0')

    @staticmethod
    def validate_num_points_left(value: int):
        if value < 0:
            raise ValueError('num_points_left must be at least 0')

    @staticmethod
    def validate_num_points_right(value: int):
        if value < 0:
            raise ValueError('num_points_right must be at least 0')

    def reset(self):
        super().reset()
        self.coefficients = None

    def configure(self, data: Matrix):
        if self.coefficients is None:
            self.coefficients = factory.create([determine_coefficients(self.num_points_left,
                                                                       self.num_points_right,
                                                                       self.polynomial_order,
                                                                       self.derivative_order)])
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        smoothed_columns = []

        window_width = self.coefficients.num_columns()
        num_output_columns = data.num_columns() - window_width + 1
        for i in range(num_output_columns):
            column = data.get_sub_matrix((0, data.num_rows()), (i, i + window_width))
            column.mul_by_vector_modify(self.coefficients)
            column = column.sum(1)
            smoothed_columns.append(column)

        return helper.multi_concat(1, *smoothed_columns)

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        pass


def determine_coefficients(num_left: int, num_right: int, poly_order: int, der_order: int) -> List[real]:
    result: List[real] = [real(0)] * (num_left + num_right + 1)

    if len(result) == 1:
        result[0] = real(1)
        return result

    A_dim = poly_order + 1
    A = factory.zeros(A_dim, A_dim)
    for i in range(A_dim):
        for j in range(A_dim):
            sum = real(0)
            for k in range(-num_left, num_right + 1):
                sum += np.power(k, i + j)
            A.set(i, j, sum)

    b = factory.zeros(A_dim, 1)
    b.set(der_order, 0, real(1))

    solution = helper.solve(A, b)

    for i in range(-num_left, num_right + 1):
        sum = real(0)
        for j in range(A_dim):
            sum += solution.get(j, 0) * np.power(i, j)
        result[i + num_left] = sum

    return result
