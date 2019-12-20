#  _DIPLS.py
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
from typing import Optional, List

from wai.common import switch, break_, case

from ._AbstractSingleResponsePLS import AbstractSingleResponsePLS
from ...core import ZERO, real, ONE, NAN
from ...core.matrix import Matrix, factory
from .._Center import Center


class DIPLS(AbstractSingleResponsePLS):
    def __init__(self):
        super().__init__()
        self.model_adaption_strategy: ModelAdaptionStrategy = ModelAdaptionStrategy.UNSUPERVISED  # Model adaption strategy
        self.ns: int = 0  # Number of source train samples
        self.nt: int = 0  # Number of target train samples
        self._lambda: real = ONE  # Lambda parameter
        self.b_0: real = ZERO  # Response mean
        self.T: Optional[Matrix] = None  # Loadings
        self.T_s: Optional[Matrix] = None  # Source domain loadings
        self.T_t: Optional[Matrix] = None  # Target domain loadings
        self.P: Optional[Matrix] = None  # Scores
        self.P_s: Optional[Matrix] = None  # Source domain scores
        self.P_t: Optional[Matrix] = None  # Target domain scores
        self.W_di: Optional[Matrix] = None  # Weights
        self.b_di: Optional[Matrix] = None  # Regression coefficients
        self.X_center: Center = Center()  # X center
        self.X_s_center: Center = Center()  # Source domain center
        self.X_t_center: Center = Center()  # Target domain center

    def get_lambda(self) -> real:
        return self._lambda

    def set_lambda(self, value: real):
        if abs(value) < 1e-8:
            raise ValueError(f"Lambda must not be zero but was {value}")

        self._lambda = value
        self.reset()

    lambda_ = property(get_lambda, set_lambda)

    def _do_reset(self):
        super()._do_reset()
        self.T = None
        self.T_s = None
        self.T_t = None
        self.P = None
        self.P_s = None
        self.P_t = None
        self.W_di = None
        self.b_di = None
        self.b_0 = NAN

    def get_matrix_names(self) -> List[str]:
        return ['T', 'Ts', 'Tt',
                'Wdi',
                'P', 'Ps', 'Pt',
                'bdi']

    def get_matrix(self, name: str) -> Optional[Matrix]:
        map = {'T': self.T,
               'Ts': self.T_s,
               'Tt': self.T_t,
               'P': self.P,
               'Ps': self.P_s,
               'Pt': self.P_t,
               'Wdi': self.W_di,
               'bdi': self.b_di}

        if name in map:
            return map[name]

        return None

    def has_loadings(self) -> bool:
        return True

    def get_loadings(self) -> Optional[Matrix]:
        return self.T

    def can_predict(self) -> bool:
        return True

    def _do_pls_transform(self, predictors: Matrix) -> Matrix:
        return self.X_center.transform(predictors).matrix_multiply(self.W_di)

    def _do_pls_configure(self, predictors: Matrix, response: Matrix):
        num_features: int = predictors.num_columns()
        I: Matrix = factory.eye(num_features)
        c: Optional[Matrix] = None

        # Check if correct initialization method was called
        if self.ns == 0 or self.nt == 0:
            return 'DIPLS must be initialized with one of the three following methods:\n' +\
                   ' - initializeSupervised' +\
                   ' - initializeSemiSupervised' +\
                   ' - initializeUnsupervisedSupervised'

        # Check if sufficient source and target samples exist
        if self.ns == 1 or self.nt == 1:
            return 'Number of source and target samples has to be > 1.'

        # Initialise Xs, Xt, X, y
        with switch(self.model_adaption_strategy):
            if case(ModelAdaptionStrategy.UNSUPERVISED):
                X_s: Matrix = predictors.get_rows((0, self.ns))
                X_t: Matrix = predictors.get_rows((self.ns, predictors.num_rows()))

                X: Matrix = X_s.copy()
                y = response
                break_()
            if case(ModelAdaptionStrategy.SUPERVISED):
                X_s: Matrix = predictors.get_rows((0, self.ns))
                X_t: Matrix = predictors.get_rows((self.ns, predictors.num_rows()))

                X: Matrix = predictors
                y = response
                break_()
            if case(ModelAdaptionStrategy.SEMISUPERVISED):
                X_s: Matrix = predictors.get_rows((0, self.ns))
                X_t: Matrix = predictors.get_rows((self.ns, predictors.num_rows()))

                X: Matrix = predictors.get_rows((0, 2 * self.ns))
                y = response
                break_()

        # Center X, Xs, Xt
        X = self.X_center.configure_and_transform(X)
        X_s = self.X_s_center.configure_and_transform(X_s)
        X_t = self.X_t_center.configure_and_transform(X_t)

        # Center y
        self.b_0 = y.mean().as_scalar()
        y = y.subtract(self.b_0)

        # Start loop over number of components
        for a in range(self._num_components):

            # Calculate domain invariant weights
            y_norm2_squared: real = y.norm2_squared()
            w_di_LHS: Matrix = y.transpose().matrix_multiply(X).divide(y_norm2_squared)
            X_s_t_X_s: Matrix = X_s.transpose().matrix_multiply(X_s).multiply(ONE / (self.ns - ONE))
            X_t_t_X_t: Matrix = X_t.transpose().matrix_multiply(X_t).multiply(ONE / (self.nt - ONE))
            X_s_diff_X_t: Matrix = X_s_t_X_s.subtract(X_t_t_X_t)
            w_di_RHS: Matrix = I.add(X_s_diff_X_t.multiply(self._lambda / (2 * y_norm2_squared))).inverse()
            w_di: Matrix = w_di_LHS.matrix_multiply(w_di_RHS).transpose()
            w_di = w_di.normalized()

            # Calculate loadings
            t: Matrix = X.matrix_multiply(w_di)
            t_s: Matrix = X_s.matrix_multiply(w_di)
            t_t: Matrix = X_t.matrix_multiply(w_di)

            # Calculate scores
            p: Matrix = (t.transpose().matrix_multiply(t)).inverse().matrix_multiply(t.transpose()).matrix_multiply(X)
            p_s: Matrix = (t_s.transpose().matrix_multiply(t_s)).inverse().matrix_multiply(t_s.transpose()).matrix_multiply(X_s)
            p_t: Matrix = (t_t.transpose().matrix_multiply(t_t)).inverse().matrix_multiply(t_t.transpose()).matrix_multiply(X_t)
            ca: Matrix = (t.transpose().matrix_multiply(t)).inverse().matrix_multiply(y.transpose()).matrix_multiply(t)

            # Deflate X, Xs, Xt, y
            X = X.subtract(t.matrix_multiply(p))
            X_s = X_s.subtract(t_s.matrix_multiply(p_s))
            X_t = X_t.subtract(t_t.matrix_multiply(p_t))
            y = y.subtract(t.matrix_multiply(ca))

            # Collect
            c = self.concat(c, ca)

            self.T = self.concat(self.T, t)
            self.T_s = self.concat(self.T_s, t_s)
            self.T_t = self.concat(self.T_t, t_t)

            self.P = self.concat(self.P, p.transpose())
            self.P_s = self.concat(self.P_s, p_s.transpose())
            self.P_t = self.concat(self.P_t, p_t.transpose())

            self.W_di = self.concat(self.W_di, w_di)

        # Calculate regression coefficients
        self.b_di = self.W_di.matrix_multiply((self.P.transpose().matrix_multiply(self.W_di)).inverse()).matrix_multiply(c.transpose())

    def concat(self, A: Optional[Matrix], a: Matrix) -> Matrix:
        """
        Concat A along columns with a. If A is None, return a.

        :param A:   Base matrix.
        :param a:   Column vector.
        :return:    Concatenation of A and a.
        """
        if A is None:
            return a
        else:
            return A.concatenate_along_columns(a)

    def _do_pls_predict(self, predictors: Matrix) -> Matrix:
        recentered: Optional[Matrix] = None

        # Recenter
        with switch(self.model_adaption_strategy):
            if case(ModelAdaptionStrategy.UNSUPERVISED):
                recentered = self.X_t_center.transform(predictors)
                break_()
            if case(ModelAdaptionStrategy.SUPERVISED,
                    ModelAdaptionStrategy.SEMISUPERVISED):
                recentered = self.X_center.transform(predictors)
                break_()

        # Predict
        regression: Matrix = recentered.matrix_multiply(self.b_di)

        # Add response means
        result: Matrix = regression.add(self.b_0)
        return result

    def configure_unsupervised(self,
                               predictors_source_domain: Matrix,
                               predictors_target_domain: Matrix,
                               response_source_domain: Matrix):
        """
        Unsupervised initialisation.

        :param predictors_source_domain:    Predictors from source domain.
        :param predictors_target_domain:    Predictors from target domain.
        :param response_source_domain:      Response from source domain.
        :return:                            Result, None if no errors, else error string.
        """
        self.ns = predictors_source_domain.num_rows()
        self.nt = predictors_target_domain.num_rows()
        self.model_adaption_strategy = ModelAdaptionStrategy.UNSUPERVISED
        X: Matrix = predictors_source_domain.concatenate_along_rows(predictors_target_domain)
        y: Matrix = response_source_domain

        return self.configure(X, y)

    def configure_supervised(self,
                             predictors_source_domain: Matrix,
                             predictors_target_domain: Matrix,
                             response_source_domain: Matrix,
                             response_target_domain: Matrix) -> str:
        """
        Supervised initialisation.

        :param predictors_source_domain:    Predictors from source domain.
        :param predictors_target_domain:    Predictors from target domain.
        :param response_source_domain:      Response from source domain.
        :param response_target_domain:      Response from target domain.
        :return:                            Result, None if no errors, else error string.
        """
        self.ns = predictors_source_domain.num_rows()
        self.nt = predictors_target_domain.num_rows()
        self.model_adaption_strategy = ModelAdaptionStrategy.SUPERVISED
        X: Matrix = predictors_source_domain.concatenate_along_rows(predictors_target_domain)
        y: Matrix = response_source_domain.concatenate_along_rows(response_target_domain)

        return self.configure(X, y)

    def configure_semisupervised(self,
                                 predictors_source_domain: Matrix,
                                 predictors_target_domain: Matrix,
                                 predictors_target_domain_unlabeled: Matrix,
                                 response_source_domain: Matrix,
                                 response_target_domain: Matrix) -> str:
        """
        Semisupervised initialisation.

        :param predictors_source_domain:            Predictors from source domain.
        :param predictors_target_domain:            Predictors from target domain.
        :param predictors_target_domain_unlabeled:  Predictors from target domain without labels.
        :param response_source_domain:              Response from source domain.
        :param response_target_domain:              Response from target domain.
        :return:                                    Result, None if no errors, else error string.
        """
        self.ns = predictors_source_domain.num_rows()
        self.nt = predictors_target_domain.num_rows()
        self.model_adaption_strategy = ModelAdaptionStrategy.SEMISUPERVISED
        X: Matrix = predictors_source_domain\
                    .concatenate_along_rows(predictors_target_domain)\
                    .concatenate_along_rows(predictors_target_domain_unlabeled)
        y: Matrix = response_source_domain.concatenate_along_rows(response_target_domain)

        return self.configure(X, y)


class ModelAdaptionStrategy(Enum):
    UNSUPERVISED = auto()
    SUPERVISED = auto()
    SEMISUPERVISED = auto()
