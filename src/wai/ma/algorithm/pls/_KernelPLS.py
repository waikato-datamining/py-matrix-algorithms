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

from ...transformation.kernel import AbstractKernel, RBFKernel
from ._AbstractMultiResponsePLS import AbstractMultiResponsePLS
from ...core import real, ZERO, ONE
from ...core.matrix import Matrix, factory
from ...transformation import Center


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
        self.K_orig: Optional[Matrix] = None  # Calibration data in feature space
        self.K_deflated: Optional[Matrix] = None
        self.T: Optional[Matrix] = None  # Scores on K
        self.U: Optional[Matrix] = None  # Scores on Y
        self.P: Optional[Matrix] = None  # Loadings on K
        self.Q: Optional[Matrix] = None  # Loadings on Y
        self.B_RHS: Optional[Matrix] = None  # Partial regression matrix
        self.X: Optional[Matrix] = None  # Training points
        self.kernel: Optional[AbstractKernel] = None  # Kernel for feature transformation
        self.tol: real = real(1e-6)  # Inner NIPALS loop improvement tolerance
        self.max_iter: int = 500  # Inner NIPALS loop maximum number of iterations
        self.center_X: Optional[Center] = None  # Center X transformation
        self.center_Y: Optional[Center] = None  # Center Y transformation

    def initialize(self, predictors: Optional[Matrix] = None, response: Optional[Matrix] = None) -> Optional[str]:
        if predictors is None and response is None:
            super().initialize()
            self.kernel = RBFKernel()
            self.tol = real(1e-6)
            self.max_iter = 500
            self.center_X = Center()
            self.center_Y = Center()
        else:
            return super().initialize(predictors, response)

    @staticmethod
    def validate_kernel(value: AbstractKernel) -> bool:
        return True

    @staticmethod
    def validate_max_iter(value: int) -> bool:
        return value >= 0

    @staticmethod
    def validate_tol(value: real) -> bool:
        return value >= ZERO

    def get_min_columns_response(self) -> int:
        return 1

    def get_max_columns_response(self) -> int:
        return -1

    def do_perform_initialization(self, predictors: Matrix, response: Matrix) -> Optional[str]:
        # Init
        num_components: int = self.num_components
        self.X = predictors
        self.X = self.center_X.transform(self.X)
        Y: Matrix = response
        Y = self.center_Y.transform(Y)

        num_rows: int = self.X.num_rows()
        num_classes: int = Y.num_columns()

        q: Matrix = factory.zeros(num_classes, 1)
        t: Matrix = factory.zeros(num_rows, 1)
        w: Matrix = factory.zeros(num_rows, 1)
        I: Matrix = factory.eye(num_rows, num_rows)

        self.T = factory.zeros(num_rows, num_components)
        self.U = factory.zeros(num_rows, num_components)
        self.P = factory.zeros(num_rows, num_components)
        self.Q = factory.zeros(num_classes, num_components)

        self.K_orig = self.kernel.apply_matrix(self.X)
        self.K_orig = self.centralize_train_in_kernel_space(self.K_orig)
        self.K_deflated = self.K_orig.copy()

        for current_component in range(num_components):
            iterations: int = 0
            u_old: Optional[Matrix] = None
            u: Matrix = factory.randn(num_rows, 1, KernelPLS.SEED + current_component)
            iteration_change: real = real(self.tol * 10)

            # Repeat 1) - 3) until convergence: either change of u is lower than self.tol or maximum
            # number of iterations has been reached (self.max_iter)
            while iteration_change > self.tol and iterations < self.max_iter:
                # 1)
                t: Matrix = self.K_deflated.mul(u).normalized()
                w: Matrix = t.copy()

                # 2)
                q = Y.transpose().mul(t)

                # 3)
                u_old = u
                u = Y.mul(q).normalized()

                # Update stopping conditions
                iterations += 1
                iteration_change = u.sub(u_old).norm2()

            # Deflate
            t_t_trans: Matrix = t.mul(t.transpose())
            part: Matrix = I.sub(t_t_trans)

            self.K_deflated = part.mul(self.K_deflated).mul(part)
            Y = Y.sub(t.mul(q.transpose()))
            p: Matrix = self.K_deflated.transpose().mul(w).div(w.transpose().mul(w).as_real())

            # Store u,t,q,p
            self.T.set_column(current_component, t)
            self.U.set_column(current_component, u)
            self.Q.set_column(current_component, q)
            self.P.set_column(current_component, p)

        # Calculate right hand side of the regression matrix B
        tT_times_K_times_U = self.T.transpose().mul(self.K_orig).mul(self.U)
        inv: Matrix = tT_times_K_times_U.inverse()
        self.B_RHS = inv.mul(self.Q.transpose())
        return None

    def centralize_train_in_kernel_space(self, K: Matrix) -> Matrix:
        """
        Centralize a kernel matrix in the kernel space via:
        K <- (I - 1/n * 1_n * 1_n^T) * K * (I - 1/n * 1_n * 1_n^T)

        :param K:   Kernel matrix.
        :return:    Centralised kernel matrix.
        """
        n: int = self.X.num_rows()
        I: Matrix = factory.eye(n, n)
        one: Matrix = factory.filled(n, 1, ONE)

        # Centralize in kernel space
        part: Matrix = I.sub(one.mul(one.transpose()).div(n))
        return part.mul(K).mul(part)

    def centralize_test_in_kernel_space(self, K: Matrix) -> Matrix:
        """
        :param K:   Kernel matrix.
        :return:    Centralised kernel matrix.
        """
        n_train: int = self.X.num_rows()
        n_test: int = K.num_rows()
        I: Matrix = factory.eye(n_train, n_train)
        ones_train_test_scaled: Matrix = factory.filled(n_test, n_train, real(ONE / n_train))

        ones_train_scaled = factory.filled(n_train, n_train, real(ONE / n_train))
        return (K.sub(ones_train_test_scaled.mul(self.K_orig))).mul(I.sub(ones_train_scaled))

    def do_perform_predictions(self, predictors: Matrix) -> Matrix:
        K_t: Matrix = self.do_transform(predictors)
        Y_hat: Matrix = K_t.mul(self.B_RHS)
        Y_hat = self.center_Y.inverse_transform(Y_hat)
        return Y_hat

    def do_transform(self, predictors: Matrix) -> Matrix:
        predictors_centered = self.center_X.transform(predictors)
        K_t: Matrix = self.kernel.apply_matrix(predictors_centered, self.X)
        K_t = self.centralize_test_in_kernel_space(K_t)

        return K_t.mul(self.U)

    def get_matrix_names(self) -> List[str]:
        return ['K', 'T', 'U', 'P', 'Q']

    def get_matrix(self, name: str) -> Optional[Matrix]:
        if name == 'K':
            return self.K_deflated
        elif name == 'T':
            return self.T
        elif name == 'U':
            return self.U
        elif name == 'P':
            return self.P
        elif name == 'Q':
            return self.Q
        return None

    def has_loadings(self) -> bool:
        return True

    def reset(self):
        super().reset()
        self.K_orig = None
        self.K_deflated = None
        self.T = None
        self.U = None
        self.P = None
        self.Q = None
        self.B_RHS = None
        self.X = None
        self.center_X = Center()
        self.center_Y = Center()

    def get_loadings(self) -> Optional[Matrix]:
        return self.T

    def can_predict(self) -> bool:
        return True
