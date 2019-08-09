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

from ...core import Filter
from .._AbstractAlgorithm import AbstractAlgorithm
from .approxfun import NegEntropyApproximationFunction, LogCosH
from ...core.error import MatrixAlgorithmsError
from ...core.matrix import Matrix
from ...core.matrix.factory import randn, zeros
from ...transformation import Center


class FastICA(AbstractAlgorithm, Filter):
    def __init__(self):
        super().__init__()
        self.num_components: int = 5
        self.whiten: bool = True
        self.fun: Optional[NegEntropyApproximationFunction] = None
        self.max_iter: int = 500
        self.tol: float = 1e-4
        self.components: Optional[Matrix] = None
        self.sources: Optional[Matrix] = None
        self.algorithm: Optional[Algorithm] = None
        self.center: Optional[Center] = None
        self.whitening: Optional[Matrix] = None
        self.mixing: Optional[Matrix] = None

    @staticmethod
    def validate_num_components(value: int) -> bool:
        return value >= 1

    @staticmethod
    def validate_whiten(value: bool) -> bool:
        return True

    @staticmethod
    def validate_max_iter(value: int) -> bool:
        return value >= 0

    @staticmethod
    def validate_tol(value: float) -> bool:
        return value >= 0.0

    def initialize(self):
        super().initialize()
        self.num_components = 5
        self.max_iter = 500
        self.fun = LogCosH()
        self.tol = 1e-4
        self.whiten = True
        self.algorithm = Algorithm.DEFLATION
        self.center = Center()

    def reset(self):
        super().reset()
        self.components = None
        self.mixing = None
        self.whitening = None

    def configure(self, X: Matrix):
        X = X.transpose()

        n: int = X.num_rows()
        p: int = X.num_columns()
        unmixing: Optional[Matrix] = None
        X1: Optional[Matrix] = None

        min_NP: int = min(n, p)

        if not self.whiten:
            self.num_components = min_NP
            self.logger.warning('Ignoring num_components when $whiten=False')

        if self.num_components > min_NP:
            self.logger.warning('num_components is too large and will be set to ' + str(min_NP))
            self.num_components = min_NP

        # Whiten data
        if self.whiten:
            X = self.center.transform(X.transpose()).transpose()
            U: Matrix = X.svd_U()
            d: Matrix = X.get_singular_values()
            k: int = min_NP  # Rank k
            d = d.get_rows((0, k))  # Only get non-zero singular values
            d_inv_elements: Matrix = d.apply_elementwise(lambda a: 1.0 / a)
            tmp: Matrix = U.scale_by_row_vector(d_inv_elements).transpose()
            self.whitening = tmp.get_rows((0, min(tmp.num_rows(), self.num_components)))

            X1 = self.whitening.mul(X)
            X1 = X1.mul(sqrt(p))
        else:
            X1 = X

        # Randomly initialise weights from normal dist
        W_init: Matrix = randn(self.num_components, self.num_components, 1)

        # Use deflation algorithm
        if self.algorithm is Algorithm.DEFLATION:
            unmixing = self.deflation(X1, W_init)
        elif self.algorithm is Algorithm.PARALLEL:  # Use parallel algorithm
            unmixing = self.parallel(X1, W_init)

        # Compute sources and components
        if self.whiten:
            self.sources = unmixing.mul(self.whitening).mul(X).transpose()
            self.components = unmixing.mul(self.whitening)
        else:
            self.sources = unmixing.mul(X).transpose()
            self.components = unmixing

        self.mixing = self.components.inverse()
        self.initialised = True

    def deflation(self, X: Matrix, W_init: Matrix) -> Matrix:
        """
        Deflationary FastICA.

        :param X:       Input.
        :return:        Weights
        """
        W: Matrix = zeros(self.num_components, self.num_components)

        for j in range(self.num_components):
            w: Matrix = W_init.get_row(j).transpose().copy()
            w = w.div(w.pow_elementwise(2).sum(-1).sqrt().as_real())
            for i in range(self.max_iter):
                res: Tuple[Matrix, Matrix] = self.fun.apply(w.transpose().mul(X).transpose())

                gwtx: Matrix = res[0]
                g_wtx: Matrix = res[1]

                w1: Matrix = X.scale_by_row_vector(gwtx).mean(1).sub(w.mul(g_wtx.mean()))
                w1 = self.decorrelate(w1, W, j)

                w1 = w1.div(w1.pow_elementwise(2).sum(-1).sqrt().as_real())
                lim = w1.mul_elementwise(w).sum(-1).abs().sub(1.0).abs().as_real()

                w = w1
                if lim < self.tol:
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

        for i in range(self.max_iter):
            res: Tuple[Matrix, Matrix] = self.fun.apply(W.transpose().mul(X))
            gwtx: Matrix = res[0]
            g_wtx: Matrix = res[1]

            arg: Matrix = gwtx.mul(X.transpose()).div(p).sub(W.scale_by_column_vector(g_wtx))  # Scale by row?
            W1: Matrix = self.symmetric_decorrelation(arg)
            lim = W1.mul(W.transpose()).diag().abs().sub(1.0).abs().max()
            W = W1
            if lim < self.tol:
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
            s: Matrix = w.transpose().mul(Wp.transpose()).mul(Wp).transpose()
            return w.sub(s)

        sub: Matrix = w.transpose().mul(Wp.transpose()).mul(Wp).transpose()
        return w.sub(sub)

    def symmetric_decorrelation(self, W: Matrix) -> Matrix:
        """
        W = (W * W.T)^(-1/2)

        :param W:   Weight matrix.
        :return:    Decorrelated weight matrix.
        """
        wwt: Matrix = W.mul(W.transpose())
        eigvals: Matrix = wwt.get_eigenvalues_sorted_ascending()
        eigvecs: Matrix = wwt.get_eigenvectors_sorted_ascending()
        s: Matrix = eigvals
        u: Matrix = eigvecs

        # np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)
        s_sqrt: Matrix = s.sqrt()
        s_inv: Matrix = s_sqrt.apply_elementwise(lambda x: 1.0 / x)
        u_mule_s: Matrix = u.scale_by_row_vector(s_inv)
        return u_mule_s.mul(u.transpose()).mul(W)

    def to_string(self) -> str:
        return 'FastICA{' +\
               'num_components=' + str(self.num_components) +\
               ', whiten=' + str(self.whiten) +\
               ', fun=' + str(self.fun) +\
               ', max_iter=' + str(self.max_iter) +\
               ', tol=' + str(self.tol) +\
               '}'

    def do_transform(self, data: Matrix) -> Matrix:
        """
        Transform a matrix.

        :param data:    The original data to transform.
        :return:        The transformed data.
        """
        return self.sources

    def transform(self, data: Matrix) -> Matrix:
        self.reset()
        self.configure(data)
        result: Matrix = self.do_transform(data)

        return result

    def reconstruct(self) -> Matrix:
        if self.is_initialised():
            return self.center.inverse_transform(self.sources.mul(self.mixing.transpose()).transpose()).transpose()
        else:
            raise MatrixAlgorithmsError('FastICA has not yet been initialized!')


class Algorithm(Enum):
    PARALLEL = auto(),
    DEFLATION = auto()
