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
        self.num_coefficients: int = 0  # The number of coefficients in W to keep (0 keep all)
        self.W: Optional[Matrix] = None  # The W matrix
        self.B: Optional[Matrix] = None  # The B matrix (used for prediction)
        self.Q: Optional[Matrix] = None  # Q matrix to regress T (XW) on y

    def initialize(self, predictors: Optional[Matrix] = None, response: Optional[Matrix] = None) -> Optional[str]:
        """
        Initialises the members.
        """
        if predictors is None and response is None:
            super().initialize()
            self.num_coefficients = 0
        else:
            return super().initialize(predictors, response)

    def reset(self):
        """
        Resets the member variables.
        """
        super().reset()
        self.B = None
        self.W = None

    @staticmethod
    def validate_num_coefficients(value: int) -> bool:
        return True

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
        B: List[List[real]] = in_.to_raw_copy_2D()

        for i in range(in_.num_columns()):
            l: Matrix = in_.get_column(i)
            ld: List[real] = l.to_raw_copy_1D()
            ld = [abs(x) for x in ld]
            srt: List[int] = utils.sort(ld)
            index: int = srt[max(len(srt) - 1 - self.num_coefficients, 0)]

            val: real = ld[index]
            for c in range(in_.num_rows()):
                if abs(B[c][i]) < val:
                    in_.set(c, i, 0)

    def do_perform_initialization(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        """
        Initializes using the provided data.

        :param predictors:  The input data.
        :param response:    The dependent variable(s).
        :return:            None if successful, otherwise error message.
        """
        X_trans: Matrix = predictors.transpose()
        A: Matrix = X_trans.mul(response)
        M: Matrix = X_trans.mul(predictors)
        C: Matrix = factory.eye(predictors.num_columns(), predictors.num_columns())
        W: Matrix = factory.zeros(predictors.num_columns(), self.num_components)
        P: Matrix = factory.zeros(predictors.num_columns(), self.num_components)
        Q: Matrix = factory.zeros(1, self.num_components)

        for h in range(self.num_components):
            # // 1. qh as dominant EigenVector of Ah'*Ah
            A_trans: Matrix = A.transpose()
            q: Matrix = A_trans.mul(A).get_dominant_eigenvector()

            # 2. wh=Ah*qh, ch=wh'*Mh*wh, wh=wh/sqrt(ch), store wh in W as column
            w: Matrix = A.mul(q)
            c: Matrix = w.transpose().mul(M).mul(w)
            w = w.mul(1.0 / sqrt(c.as_real()))
            W.set_column(h, w)

            # 3. ph=Mh*wh, store ph in P as column
            p: Matrix = M.mul(w)
            p_trans: Matrix = p.transpose()
            P.set_column(h, p)

            # 4. qh=Ah'*wh, store qh in Q as column
            q: Matrix = A_trans.mul(w)
            Q.set_column(h, q)

            # 5. vh=Ch*ph, vh=vh/||vh||
            v: Matrix = C.mul(p)
            v = v.normalized()
            v_trans: Matrix = v.transpose()

            # 6. Ch+1=Ch-vh*vh', Mh+1=Mh-ph*ph'
            C = C.sub(v.mul(v_trans))
            M = M.sub(p.mul(p_trans))

            # 7. Ah+1=ChAh (actually Ch+1)
            A = C.mul(A)

        # Finish
        if self.num_coefficients > 0:
            self.slim(W)
        self.W = W
        self.B = W.mul(Q.transpose())
        self.Q = Q

        return None

    def do_transform(self, predictors: Matrix) -> Matrix:
        """
        Transforms the data.

        :param predictors:  The input data.
        :return:            The transformed data.
        """
        return predictors.mul(self.W)

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
        :return:            The predictions.
        """
        return predictors.mul(self.B)

    def serialise_state(self, stream: IO[bytes]):
        # Can't serialise our state until we've been initialised
        if not self.initialised:
            raise RuntimeError("Can't serialise state of uninitialised SIMPLS")

        # Serialise out our W matrix
        self.W.serialise_state(stream)
