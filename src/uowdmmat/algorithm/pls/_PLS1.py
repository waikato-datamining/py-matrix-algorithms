#  _PLS1.py
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

from ._AbstractSingleResponsePLS import AbstractSingleResponsePLS
from ...core import real
from ...core.matrix import Matrix, factory, helper


class PLS1(AbstractSingleResponsePLS):
    def __init__(self):
        super().__init__()
        self.r_hat: Optional[Matrix] = None  # The regression vector "r-hat"
        self.P: Optional[Matrix] = None  # The P matrix
        self.W: Optional[Matrix] = None  # The W matrix
        self.b_hat: Optional[Matrix] = None  # The b-hat vector

    def reset(self):
        """
        Resets the member variables.
        """
        super().reset()

        self.r_hat = None
        self.P = None
        self.W = None
        self.b_hat = None

    def get_matrix_names(self) -> List[str]:
        """
        Returns the names of all the available matrices.

        :return:    The names of the matrices.
        """
        return ['RegVector',
                'P',
                'W',
                'b_hat']

    def get_matrix(self, name: str) -> Optional[Matrix]:
        """
        Returns the matrix with the specified name.

        :param name:    The name of the matrix.
        :return:        The matrix, None if not available.
        """
        if name == 'RegVector':
            return self.r_hat
        elif name == 'P':
            return self.P
        elif name == 'W':
            return self.W
        elif name == 'b_hat':
            return self.b_hat
        else:
            return None

    def has_loadings(self) -> bool:
        """
        Whether the algorithm supports the return of loadings.

        :return:    True if supported.
        """
        return True

    def get_loadings(self) -> Optional[Matrix]:
        """
        Returns the loadings, if available.

        :return:     The loadings, None if not available.
        """
        return self.get_matrix('P')

    def do_perform_initialization(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        """
        Initializes using the provided data.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        X_k: Matrix = predictors
        y: Matrix = response

        # Init
        W: Matrix = factory.zeros(predictors.num_columns(), self.num_components)
        P: Matrix = factory.zeros(predictors.num_columns(), self.num_components)
        T: Matrix = factory.zeros(predictors.num_rows(), self.num_components)
        b_hat: Matrix = factory.zeros(self.num_components, 1)

        for k in range(self.num_components):
            # 1. step: wj
            w_k: Matrix = self.calculate_weights(X_k, y)
            W.set_column(k, w_k)

            # 2. step: tj
            t_k: Matrix = X_k.mul(w_k)
            T.set_column(k, t_k)

            # 3. step: ^bj
            tdott: real = t_k.vector_dot(t_k)
            b_k: real = real(t_k.vector_dot(y) / tdott)
            b_hat.set(k, 0, b_k)

            # 4. step: pj
            p_k: Matrix = X_k.transpose().mul(t_k).div(tdott)
            P.set_column(k, p_k)

            # 5. step: Xk+1 (deflating y is not necessary)
            X_k = X_k.sub(t_k.mul(p_k.transpose()))

        # W*(P^T*W)^-1
        tmp: Matrix = W.mul(((P.transpose()).mul(W)).inverse())

        # factor = W*(P^T*W)^-1 * b_hat
        self.r_hat = tmp.mul(b_hat)

        # Save matrices
        self.P = P
        self.W = W
        self.b_hat = b_hat

        return None

    def calculate_weights(self, x_k: Matrix, y: Matrix) -> Matrix:
        """
        Calculate the weight w_k in the PLS iterations.

        :param x_k:     X matrix at step k.
        :param y:       y matrix.
        :return:        Weights at step k.
        """
        return x_k.transpose().mul(y).normalized()

    def do_transform(self, predictors: Matrix) -> Matrix:
        """
        Transforms the data.

        :param predictors:  The input data.
        :return:            The transformed data and the predictions.
        """
        result = factory.zeros(predictors.num_rows(), self.num_components)

        for i in range(predictors.num_rows()):
            # Work on each row
            x: Matrix = helper.row_as_vector(predictors, i)
            X: Matrix = factory.zeros(1, self.num_components)
            T: Matrix = factory.zeros(1, self.num_components)

            for j in range(self.num_components):
                X.set_column(j, x)
                # 1. step: tj = xj * wj
                t: Matrix = x.mul(self.W.get_column(j))
                T.set_column(j, t)
                # 2. step:xj+1 = xj - tj*pj^T (tj is 1x1 matrix!)
                x = x.sub(self.P.get_column(j).transpose().mul(t.as_real()))

            result.set_row(i, T)

        return result

    def can_predict(self) -> bool:
        """
        Returns whether the algorithm can make predictions.

        :return:    True if can make predictions.
        """
        return True

    def do_perform_predictions(self, predictors: Matrix) -> Matrix:
        """
        Performs predictions on the data.

        :param predictors:  The input data.
        :return:            The transformed data and the predictions.
        """
        result = factory.zeros(predictors.num_rows(), 1)

        for i in range(predictors.num_rows()):
            x: Matrix = helper.row_as_vector(predictors, i)
            X: Matrix = factory.zeros(1, self.num_components)
            T: Matrix = factory.zeros(1, self.num_components)

            for j in range(self.num_components):
                X.set_column(j, x)
                # 1. step: tj = xj * wj
                t: Matrix = x.mul(self.W.get_column(j))
                T.set_column(j, t)
                # 2. step: xj+1 = xj - tj*pj^T (tj is 1x1 matrix!)
                x = x.sub(self.P.get_column(j).transpose().mul(t.as_real()))

            result.set(i, 0, T.mul(self.b_hat).as_real())

        return result
