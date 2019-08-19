#  _DIPLSTest.py
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
from wai.test.decorators import RegressionTest, Test

from wai.ma.algorithm.pls import DIPLS
from wai.ma.core.matrix import Matrix
from wai.ma.core.matrix.factory import randn_like

from ._AbstractPLSTest import AbstractPLSTest
from ...test import Tags


class DIPLSTest(AbstractPLSTest):
    @classmethod
    def subject_type(cls):
        return DIPLS

    @RegressionTest
    def lambda_01(self, subject: DIPLS, bolts: Matrix, bolts_response: Matrix):
        subject.lambda_ = 0.01
        return self.standard_regression(subject, bolts, bolts_response)

    def standard_regression(self, subject: DIPLS, *resources: Matrix):
        x_source_domain, y_source_domain = resources
        x_target_domain: Matrix = x_source_domain.add(randn_like(x_source_domain, 0, 1, 2))
        y_target_domain: Matrix = y_source_domain.add(randn_like(y_source_domain, 1, 1, 2))
        x_target_domain_unlabeled: Matrix = x_source_domain.add(randn_like(x_source_domain, 100, 1, 2))

        result = {}

        # Initialise supervised
        subject.initialize_supervised(x_source_domain, x_target_domain, y_source_domain, y_target_domain)
        result.update(self.add_default_pls_matrices(subject, x_target_domain, Tags.SUPERVISED))
        subject.reset()

        # Initialise unsupervised
        subject.initialize_unsupervised(x_source_domain, x_target_domain, y_source_domain)
        result.update(self.add_default_pls_matrices(subject, x_target_domain, Tags.UNSUPERVISED))
        subject.reset()

        # Initialise semisupervised
        subject.initialize_semisupervised(x_source_domain, x_target_domain, x_target_domain_unlabeled,
                                          y_source_domain, y_target_domain)
        result.update(self.add_default_pls_matrices(subject, x_target_domain, Tags.SEMISUPERVISED))

        return result

    @Test
    def check_transformed_num_components(self, subject: DIPLS, bolts: Matrix, bolts_response: Matrix):
        X, Y = bolts, bolts_response
        X_2: Matrix = X.add(randn_like(X, 0, 0, 2))

        for i in range(1, 5):
            subject.num_components = i
            subject.initialize_unsupervised(X, X_2, Y)
            transform: Matrix = subject.transform(X)
            self.assertEqual(i, transform.num_columns())

            # Reset
            subject = self.instantiate_subject()
