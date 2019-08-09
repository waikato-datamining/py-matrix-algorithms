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
from ...transformation import Center


class DIPLS(AbstractSingleResponsePLS):
    def __init__(self):
        super().__init__()
        self.model_adaption_strategy: Optional[ModelAdaptionStrategy] = None  # Model adaption strategy
        self.ns: int = 0  # Number of source train samples
        self.nt: int = 0  # Number of target train samples
        self.lambda_: real = ONE  # Lambda parameter
        self.b_0: real = ZERO  # Response mean
        self.T: Optional[Matrix] = None  # Loadings
        self.T_s: Optional[Matrix] = None  # Source domain loadings
        self.T_t: Optional[Matrix] = None  # Target domain loadings
        self.P: Optional[Matrix] = None  # Scores
        self.P_s: Optional[Matrix] = None  # Source domain scores
        self.P_t: Optional[Matrix] = None  # Target domain scores
        self.W_di: Optional[Matrix] = None  # Weights
        self.b_di: Optional[Matrix] = None  # Regression coefficients
        self.X_center: Optional[Center] = None  # X center
        self.X_s_center: Optional[Center] = None  # Source domain center
        self.X_t_center: Optional[Center] = None  # Target domain center

    def initialize(self, predictors: Optional[Matrix] = None, response: Optional[Matrix] = None) -> Optional[str]:
        if predictors is None and response is None:
            super().initialize()
            self.model_adaption_strategy = ModelAdaptionStrategy.UNSUPERVISED
            self.lambda_ = ONE
            self.X_center = Center()
            self.X_t_center = Center()
            self.X_s_center = Center()
        else:
            return super().initialize(predictors, response)

    def reset(self):
        super().reset()
        self.T = None
        self.T_s = None
        self.T_t = None
        self.P = None
        self.P_s = None
        self.P_t = None
        self.W_di = None
        self.b_di = None
        self.b_0 = NAN

    @staticmethod
    def validate_lambda_(value: real) -> bool:
        return abs(value) >= 1e-8

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

    def do_transform(self, predictors: Matrix) -> Matrix:
        return self.X_center.transform(predictors).mul(self.W_di)

    def do_perform_initialization(self, predictors: Matrix, response: Matrix) -> Optional[str]:
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
        X = self.X_center.transform(X)
        X_s = self.X_s_center.transform(X_s)
        X_t = self.X_t_center.transform(X_t)

        # Center y
        self.b_0 = y.mean(-1).as_real()
        y = y.sub(self.b_0)

        # Start loop over number of components
        for a in range(self.num_components):

            # Calculate domain invariant weights
            y_norm2_squared: real = y.norm2_squared()
            w_di_LHS: Matrix = y.transpose().mul(X).div(y_norm2_squared)
            X_s_t_X_s: Matrix = X_s.transpose().mul(X_s).mul(ONE / (self.ns - ONE))
            X_t_t_X_t: Matrix = X_t.transpose().mul(X_t).mul(ONE / (self.nt - ONE))
            X_s_diff_X_t: Matrix = X_s_t_X_s.sub(X_t_t_X_t)
            w_di_RHS: Matrix = I.add(X_s_diff_X_t.mul(self.lambda_ / (2 * y_norm2_squared))).inverse()
            w_di: Matrix = w_di_LHS.mul(w_di_RHS).transpose()
            w_di = w_di.normalized()

            # Calculate loadings
            t: Matrix = X.mul(w_di)
            t_s: Matrix = X_s.mul(w_di)
            t_t: Matrix = X_t.mul(w_di)

            # Calculate scores
            p: Matrix = (t.transpose().mul(t)).inverse().mul(t.transpose()).mul(X)
            p_s: Matrix = (t_s.transpose().mul(t_s)).inverse().mul(t_s.transpose()).mul(X_s)
            p_t: Matrix = (t_t.transpose().mul(t_t)).inverse().mul(t_t.transpose()).mul(X_t)
            ca: Matrix = (t.transpose().mul(t)).inverse().mul(y.transpose()).mul(t)

            # Deflate X, Xs, Xt, y
            X = X.sub(t.mul(p))
            X_s = X_s.sub(t_s.mul(p_s))
            X_t = X_t.sub(t_t.mul(p_t))
            y = y.sub(t.mul(ca))

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
        self.b_di = self.W_di.mul((self.P.transpose().mul(self.W_di)).inverse()).mul(c.transpose())

        return None

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
            return A.concat_along_columns(a)

    def do_perform_predictions(self, predictors: Matrix) -> Matrix:
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
        regression: Matrix = recentered.mul(self.b_di)

        # Add response means
        result: Matrix = regression.add(self.b_0)
        return result

    def initialize_unsupervised(self,
                                predictors_source_domain: Matrix,
                                predictors_target_domain: Matrix,
                                response_source_domain: Matrix) -> str:
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
        X: Matrix = predictors_source_domain.concat_along_rows(predictors_target_domain)
        y: Matrix = response_source_domain

        return self.initialize(X, y)

    def initialize_supervised(self,
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
        X: Matrix = predictors_source_domain.concat_along_rows(predictors_target_domain)
        y: Matrix = response_source_domain.concat_along_rows(response_target_domain)

        return self.initialize(X, y)

    def initialize_semisupervised(self,
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
                    .concat_along_rows(predictors_target_domain)\
                    .concat_along_rows(predictors_target_domain_unlabeled)
        y: Matrix = response_source_domain.concat_along_rows(response_target_domain)

        return self.initialize(X, y)


class ModelAdaptionStrategy(Enum):
    UNSUPERVISED = auto()
    SUPERVISED = auto()
    SEMISUPERVISED = auto()
