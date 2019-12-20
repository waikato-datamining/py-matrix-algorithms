#  _SIMPLS.py
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
from typing import Optional, List, IO

from ._AbstractSingleResponsePLS import AbstractSingleResponsePLS
from ...core import real, utils, Serialisable
from ...core.matrix import Matrix, factory
from ...core.utils import sqrt


class SIMPLS(AbstractSingleResponsePLS, Serialisable):
    def __init__(self):
        super().__init__()
        self._num_coefficients: int = 0  # The number of coefficients in W to keep (0 keep all)
        self.W: Optional[Matrix] = None  # The W matrix
        self.B: Optional[Matrix] = None  # The B matrix (used for prediction)
        self.Q: Optional[Matrix] = None  # Q matrix to regress T (XW) on y

    def _do_reset(self):
        """
        Resets the member variables.
        """
        super()._do_reset()
        self.B = None
        self.W = None

    def get_num_coefficients(self) -> int:
        return self._num_coefficients

    def set_num_coefficients(self, value: int):
        self._num_coefficients = value
        self.reset()

    num_coefficients = property(get_num_coefficients, set_num_coefficients)

    def get_matrix_names(self) -> List[str]:
        """
        Returns all available matrices.

        :return:    The names of the matrices.
        """
        return ['W', 'B', 'Q']

    def get_matrix(self, name: str) -> Optional[Matrix]:
        """
        Returns the matrix with the specified name.

        :param name:    The name of the matrix.
        :return:        The matrix, None if not available.
        """
        if name == 'W':
            return self.W
        elif name == 'B':
            return self.B
        elif name == 'Q':
            return self.Q
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

        :return:    The loadings, None if not available.
        """
        return self.W

    def slim(self, in_: Matrix):
        """
        Zeroes the coefficients of the W matrix beyond the specified number of
        coefficients.

        :param in_: The matrix to process in-place.
        """
        B: List[List[real]] = in_.as_native()

        for i in range(in_.num_columns()):
            l: Matrix = in_.get_column(i)
            ld: List[real] = l.as_native_flattened()
            ld = [abs(x) for x in ld]
            srt: List[int] = utils.sort(ld)
            index: int = srt[max(len(srt) - 1 - self._num_coefficients, 0)]

            val: real = ld[index]
            for c in range(in_.num_rows()):
                if abs(B[c][i]) < val:
                    in_.set(c, i, 0)

    def _do_pls_configure(self, predictors: Matrix, response: Matrix):
        """
        Initializes using the provided data.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        X_trans: Matrix = predictors.transpose()
        A: Matrix = X_trans.matrix_multiply(response)
        M: Matrix = X_trans.matrix_multiply(predictors)
        C: Matrix = factory.eye(predictors.num_columns(), predictors.num_columns())
        W: Matrix = factory.zeros(predictors.num_columns(), self._num_components)
        P: Matrix = factory.zeros(predictors.num_columns(), self._num_components)
        Q: Matrix = factory.zeros(1, self._num_components)

        for h in range(self._num_components):
            # // 1. qh as dominant EigenVector of Ah'*Ah
            A_trans: Matrix = A.transpose()
            q: Matrix = A_trans.matrix_multiply(A).get_dominant_eigenvector()

            # 2. wh=Ah*qh, ch=wh'*Mh*wh, wh=wh/sqrt(ch), store wh in W as column
            w: Matrix = A.matrix_multiply(q)
            c: Matrix = w.transpose().matrix_multiply(M).matrix_multiply(w)
            w = w.multiply(1.0 / sqrt(c.as_scalar()))
            W.set_column(h, w)

            # 3. ph=Mh*wh, store ph in P as column
            p: Matrix = M.matrix_multiply(w)
            p_trans: Matrix = p.transpose()
            P.set_column(h, p)

            # 4. qh=Ah'*wh, store qh in Q as column
            q: Matrix = A_trans.matrix_multiply(w)
            Q.set_column(h, q)

            # 5. vh=Ch*ph, vh=vh/||vh||
            v: Matrix = C.matrix_multiply(p)
            v = v.normalized()
            v_trans: Matrix = v.transpose()

            # 6. Ch+1=Ch-vh*vh', Mh+1=Mh-ph*ph'
            C = C.subtract(v.matrix_multiply(v_trans))
            M = M.subtract(p.matrix_multiply(p_trans))

            # 7. Ah+1=ChAh (actually Ch+1)
            A = C.matrix_multiply(A)

        # Finish
        if self._num_coefficients > 0:
            self.slim(W)
        self.W = W
        self.B = W.matrix_multiply(Q.transpose())
        self.Q = Q

        return None

    def _do_pls_transform(self, predictors: Matrix) -> Matrix:
        """
        Transforms the data.

        :param predictors:  The input data.
        :return:            The transformed data.
        """
        return predictors.matrix_multiply(self.W)

    def can_predict(self) -> bool:
        """
        Returns whether the algorithm can make predictions.

        :return:    True if can make predictions.
        """
        return True

    def _do_pls_predict(self, predictors: Matrix) -> Matrix:
        """
        Performs predictions on the data.

        :param predictors:  The input data.
        :return:            The predictions.
        """
        return predictors.matrix_multiply(self.B)

    def serialise_state(self, stream: IO[bytes]):
        # Can't serialise our state until we've been initialised
        if not self.is_configured():
            raise RuntimeError("Can't serialise state of uninitialised SIMPLS")

        # Serialise out our W matrix
        self.W.serialise_state(stream)
