#  _KernelPLS.py
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

from ...core import real, ONE
from ...core.matrix import Matrix, factory
from .._Center import Center
from .kernel import AbstractKernel, RBFKernel
from ._AbstractMultiResponsePLS import AbstractMultiResponsePLS


class KernelPLS(AbstractMultiResponsePLS):
    """
    Kernel Partial Least Squares algorithm.

    See here:
    <a href="http://www.jmlr.org/papers/volume2/rosipal01a/rosipal01a.pdf">
        Kernel Partial Least Squares Regression in Reproducing Kernel Hilbert Space
    </a>
    """
    SEED: int = 0

    def __init__(self):
        super().__init__()

        self._K_orig: Optional[Matrix] = None  # Calibration data in feature space
        self._K_deflated: Optional[Matrix] = None
        self._T: Optional[Matrix] = None  # Scores on K
        self._U: Optional[Matrix] = None  # Scores on Y
        self._P: Optional[Matrix] = None  # Loadings on K
        self._Q: Optional[Matrix] = None  # Loadings on Y
        self._B_RHS: Optional[Matrix] = None  # Partial regression matrix
        self._X: Optional[Matrix] = None  # Training points
        self._kernel: AbstractKernel = RBFKernel()  # Kernel for feature transformation
        self._tol: real = real(1e-6)  # Inner NIPALS loop improvement tolerance
        self._max_iter: int = 500  # Inner NIPALS loop maximum number of iterations
        self._center_X: Center = Center()  # Center X transformation
        self._center_Y: Center = Center()  # Center Y transformation

    def get_kernel(self) -> AbstractKernel:
        return self._kernel

    def set_kernel(self, value: AbstractKernel):
        self._kernel = value
        self.reset()

    kernel = property(get_kernel, set_kernel)

    def get_max_iter(self) -> int:
        return self._max_iter

    def set_max_iter(self, value: int):
        if value < 0:
            raise ValueError(f"Maximum iteration parameter must be positive but was {value}")

        self._max_iter = value
        self.reset()

    max_iter = property(get_max_iter, set_max_iter)

    def get_tol(self) -> real:
        return self._tol

    def set_tol(self, value: real):
        if value < 0:
            raise ValueError(f"Tolerance parameter must be positive but was {value}")

        self._tol = value
        self.reset()

    tol = property(get_tol, set_tol)

    def get_min_columns_response(self) -> int:
        return 1

    def get_max_columns_response(self) -> int:
        return -1

    def _do_pls_configure(self, predictors: Matrix, response: Matrix):
        # Init
        num_components: int = self._num_components
        self._X = predictors
        self._X = self._center_X.configure_and_transform(self._X)
        Y: Matrix = response
        Y = self._center_Y.configure_and_transform(Y)

        num_rows: int = self._X.num_rows()
        num_classes: int = Y.num_columns()

        q: Matrix = factory.zeros(num_classes, 1)
        t: Matrix = factory.zeros(num_rows, 1)
        w: Matrix = factory.zeros(num_rows, 1)
        I: Matrix = factory.eye(num_rows, num_rows)

        self._T = factory.zeros(num_rows, num_components)
        self._U = factory.zeros(num_rows, num_components)
        self._P = factory.zeros(num_rows, num_components)
        self._Q = factory.zeros(num_classes, num_components)

        self._K_orig = self._kernel.apply_matrix(self._X)
        self._K_orig = self.centralize_train_in_kernel_space(self._K_orig)
        self._K_deflated = self._K_orig.copy()

        for current_component in range(num_components):
            iterations: int = 0
            u_old: Optional[Matrix] = None
            u: Matrix = factory.randn(num_rows, 1, KernelPLS.SEED + current_component)
            iteration_change: real = real(self._tol * 10)

            # Repeat 1) - 3) until convergence: either change of u is lower than self.tol or maximum
            # number of iterations has been reached (self.max_iter)
            while iteration_change > self._tol and iterations < self._max_iter:
                # 1)
                t: Matrix = self._K_deflated.matrix_multiply(u).normalized()
                w: Matrix = t.copy()

                # 2)
                q = Y.transpose().matrix_multiply(t)

                # 3)
                u_old = u
                u = Y.matrix_multiply(q).normalized()

                # Update stopping conditions
                iterations += 1
                iteration_change = u.subtract(u_old).norm2()

            # Deflate
            t_t_trans: Matrix = t.matrix_multiply(t.transpose())
            part: Matrix = I.subtract(t_t_trans)

            self._K_deflated = part.matrix_multiply(self._K_deflated).matrix_multiply(part)
            Y = Y.subtract(t.matrix_multiply(q.transpose()))
            p: Matrix = self._K_deflated.transpose().matrix_multiply(w).divide(w.transpose().matrix_multiply(w).as_scalar())

            # Store u,t,q,p
            self._T.set_column(current_component, t)
            self._U.set_column(current_component, u)
            self._Q.set_column(current_component, q)
            self._P.set_column(current_component, p)

        # Calculate right hand side of the regression matrix B
        tT_times_K_times_U = self._T.transpose().matrix_multiply(self._K_orig).matrix_multiply(self._U)
        inv: Matrix = tT_times_K_times_U.inverse()
        self._B_RHS = inv.matrix_multiply(self._Q.transpose())

    def centralize_train_in_kernel_space(self, K: Matrix) -> Matrix:
        """
        Centralize a kernel matrix in the kernel space via:
        K <- (I - 1/n * 1_n * 1_n^T) * K * (I - 1/n * 1_n * 1_n^T)

        :param K:   Kernel matrix.
        :return:    Centralised kernel matrix.
        """
        n: int = self._X.num_rows()
        I: Matrix = factory.eye(n, n)
        one: Matrix = factory.filled(n, 1, ONE)

        # Centralize in kernel space
        part: Matrix = I.subtract(one.matrix_multiply(one.transpose()).divide(n))
        return part.matrix_multiply(K).matrix_multiply(part)

    def centralize_test_in_kernel_space(self, K: Matrix) -> Matrix:
        """
        :param K:   Kernel matrix.
        :return:    Centralised kernel matrix.
        """
        n_train: int = self._X.num_rows()
        n_test: int = K.num_rows()
        I: Matrix = factory.eye(n_train, n_train)
        ones_train_test_scaled: Matrix = factory.filled(n_test, n_train, real(ONE / n_train))

        ones_train_scaled = factory.filled(n_train, n_train, real(ONE / n_train))
        return (K.subtract(ones_train_test_scaled.matrix_multiply(self._K_orig))).matrix_multiply(I.subtract(ones_train_scaled))

    def _do_pls_predict(self, predictors: Matrix) -> Matrix:
        K_t: Matrix = self._do_pls_transform(predictors)
        Y_hat: Matrix = K_t.matrix_multiply(self._B_RHS)
        Y_hat = self._center_Y.inverse_transform(Y_hat)
        return Y_hat

    def _do_pls_transform(self, predictors: Matrix) -> Matrix:
        predictors_centered = self._center_X.transform(predictors)
        K_t: Matrix = self._kernel.apply_matrix(predictors_centered, self._X)
        K_t = self.centralize_test_in_kernel_space(K_t)

        return K_t.matrix_multiply(self._U)

    def get_matrix_names(self) -> List[str]:
        return ['K', 'T', 'U', 'P', 'Q']

    def get_matrix(self, name: str) -> Optional[Matrix]:
        if name == 'K':
            return self._K_deflated
        elif name == 'T':
            return self._T
        elif name == 'U':
            return self._U
        elif name == 'P':
            return self._P
        elif name == 'Q':
            return self._Q
        return None

    def has_loadings(self) -> bool:
        return True

    def _do_reset(self):
        super()._do_reset()
        self._K_orig = None
        self._K_deflated = None
        self._T = None
        self._U = None
        self._P = None
        self._Q = None
        self._B_RHS = None
        self._X = None
        self._center_X.reset()
        self._center_Y.reset()

    def get_loadings(self) -> Optional[Matrix]:
        return self._T

    def can_predict(self) -> bool:
        return True
