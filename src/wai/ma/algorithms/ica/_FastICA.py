#  _FastICA.py
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
from enum import Enum, auto
from typing import Optional, Tuple

from math import sqrt

from ...core.algorithm import MatrixAlgorithm
from ...core.errors import MatrixAlgorithmsError
from ...core.matrix import Matrix, Axis
from ...core.matrix.factory import randn, zeros
from .._Center import Center
from .approxfun import NegEntropyApproximationFunction, LogCosH


class FastICA(MatrixAlgorithm):
    def __init__(self):
        super().__init__()
        self._num_components: int = 5
        self.whiten: bool = True
        self.fun: NegEntropyApproximationFunction = LogCosH()
        self._max_iter: int = 500
        self._tol: float = 1e-4
        self._components: Optional[Matrix] = None
        self._sources: Optional[Matrix] = None
        self.algorithm: Optional[Algorithm] = None
        self._center: Optional[Center] = None
        self._whitening: Optional[Matrix] = None
        self._mixing: Optional[Matrix] = None

    def get_num_components(self) -> int:
        return self._num_components

    def set_num_components(self, value: int):
        if value < 1:
            raise ValueError(f"num_components must be at least 1, got {value}")

        self._num_components = value

    num_components = property(get_num_components, set_num_components)

    def get_max_iter(self) -> int:
        return self._max_iter

    def set_max_iter(self, value: int):
        if value < 0:
            raise ValueError(f"max_iter must be at least 0 but got {value}")

        self._max_iter = value

    max_iter = property(get_max_iter, set_max_iter)

    def get_tol(self) -> float:
        return self._tol

    def set_tol(self, value: float):
        if value < 0.0:
            raise ValueError(f"tol must be at least 0 but got {value}")

        self._tol = value

    tol = property(get_tol, set_tol)

    def get_components(self) -> Matrix:
        return self._components

    def get_sources(self) -> Matrix:
        return self._sources

    def get_mixing(self) -> Matrix:
        return self._mixing

    def deflation(self, X: Matrix, W_init: Matrix) -> Matrix:
        """
        Deflationary FastICA.

        :param X:       Input.
        :return:        Weights
        """
        W: Matrix = zeros(self._num_components, self._num_components)

        for j in range(self._num_components):
            w: Matrix = W_init.get_row(j).transpose()
            w = w.divide(w.pow(2).total(as_real=False).sqrt().as_scalar())
            for i in range(self._max_iter):
                res: Tuple[Matrix, Matrix] = self.fun.apply(w.transpose().matrix_multiply(X).transpose())

                gwtx: Matrix = res[0]
                g_wtx: Matrix = res[1]

                w1: Matrix = X.multiply(gwtx).mean(Axis.ROWS).sub(w.multiply(g_wtx.mean()))
                w1 = self.decorrelate(w1, W, j)

                w1 = w1.divide(w1.pow(2).total(as_real=False).sqrt().as_scalar())
                lim = w1.multiply(w).total(as_real=False).abs().subtract(1.0).abs().as_scalar()

                w = w1
                if lim < self._tol:
                    break

            W.set_row(j, w)

        return W

    def parallel(self, X: Matrix, W_init: Matrix) -> Matrix:
        """
        Parallel FastICA.

        :param X:       Input.
        :param W_init:  Initial weight matrix.
        :return:        Weight.
        """
        W: Matrix = self.symmetric_decorrelation(W_init)

        p: int = X.num_columns()

        for i in range(self._max_iter):
            res: Tuple[Matrix, Matrix] = self.fun.apply(W.transpose().matrix_multiply(X))
            gwtx: Matrix = res[0]
            g_wtx: Matrix = res[1]

            arg: Matrix = gwtx.matrix_multiply(X.transpose()).divide(p).subtract(W.multiply(g_wtx))  # Scale by row?
            W1: Matrix = self.symmetric_decorrelation(arg)
            lim = W1.matrix_multiply(W.transpose()).diag().abs().subtract(1.0).abs().maximum()
            W = W1
            if lim < self._tol:
                break

        return W

    def decorrelate(self, w: Matrix, W: Matrix, j: int) -> Matrix:
        """
        Orthonormalise w wrt the first j columns of W.

        :param w:   w vector.
        :param W:   W matrix.
        :param j:   First j columns.
        :return:    Orthonormalised w.
        """
        if j == 0:
            return w
        Wp: Matrix = W.get_rows((0, j))

        if j == 1:
            s: Matrix = w.transpose().matrix_multiply(Wp.transpose()).matrix_multiply(Wp).transpose()
            return w.subtract(s)

        sub: Matrix = w.transpose().matrix_multiply(Wp.transpose()).matrix_multiply(Wp).transpose()
        return w.subtract(sub)

    def symmetric_decorrelation(self, W: Matrix) -> Matrix:
        """
        W = (W * W.T)^(-1/2)

        :param W:   Weight matrix.
        :return:    Decorrelated weight matrix.
        """
        wwt: Matrix = W.matrix_multiply(W.transpose())
        eigvals: Matrix = wwt.get_eigenvalues_sorted_ascending()
        eigvecs: Matrix = wwt.get_eigenvectors_sorted_ascending()
        s: Matrix = eigvals
        u: Matrix = eigvecs

        # np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)
        s_sqrt: Matrix = s.sqrt()
        s_inv: Matrix = s_sqrt.apply_elementwise(lambda x: 1.0 / x)
        u_mule_s: Matrix = u.scale_by_row_vector(s_inv)
        return u_mule_s.matrix_multiply(u.transpose()).matrix_multiply(W)

    def to_string(self) -> str:
        return 'FastICA{' +\
               'num_components=' + str(self._num_components) +\
               ', whiten=' + str(self.whiten) +\
               ', fun=' + str(self.fun) +\
               ', max_iter=' + str(self._max_iter) +\
               ', tol=' + str(self._tol) +\
               '}'

    def _do_transform(self, X: Matrix) -> Matrix:
        X = X.transpose()

        n: int = X.num_rows()
        p: int = X.num_columns()
        unmixing: Optional[Matrix] = None
        X1: Optional[Matrix] = None

        min_NP: int = min(n, p)

        if not self.whiten:
            self._num_components = min_NP
            self.logger.warning('Ignoring num_components when $whiten=False')

        if self._num_components > min_NP:
            self.logger.warning('num_components is too large and will be set to ' + str(min_NP))
            self._num_components = min_NP

        # Whiten data
        if self.whiten:
            X = self._center.transform(X.transpose()).transpose()
            U: Matrix = X.svd_U()
            d: Matrix = X.get_singular_values()
            k: int = min_NP  # Rank k
            d = d.get_rows((0, k))  # Only get non-zero singular values
            d_inv_elements: Matrix = d.apply_elementwise(lambda a: 1.0 / a)
            tmp: Matrix = U.scale_by_row_vector(d_inv_elements).transpose()
            self._whitening = tmp.get_rows((0, min(tmp.num_rows(), self._num_components)))

            X1 = self._whitening.matrix_multiply(X)
            X1 = X1.matrix_multiply(sqrt(p))
        else:
            X1 = X

        # Randomly initialise weights from normal dist
        W_init: Matrix = randn(self._num_components, self._num_components, 1)

        # Use deflation algorithm
        if self.algorithm is Algorithm.DEFLATION:
            unmixing = self.deflation(X1, W_init)
        elif self.algorithm is Algorithm.PARALLEL:  # Use parallel algorithm
            unmixing = self.parallel(X1, W_init)

        # Compute sources and components
        if self.whiten:
            self._sources = unmixing.matrix_multiply(self._whitening).matrix_multiply(X).transpose()
            self._components = unmixing.matrix_multiply(self._whitening)
        else:
            self._sources = unmixing.matrix_multiply(X).transpose()
            self._components = unmixing

        self._mixing = self._components.inverse()

        return self._sources

    def reconstruct(self) -> Matrix:
        if self.is_initialised():
            return self._center.inverse_transform(self._sources.matrix_multiply(self._mixing.transpose()).transpose()).transpose()
        else:
            raise MatrixAlgorithmsError('FastICA has not yet been initialized!')

    def is_non_invertible(self) -> bool:
        return True


class Algorithm(Enum):
    PARALLEL = auto(),
    DEFLATION = auto()
