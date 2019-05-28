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
from typing import List

from ._AbstractPLSTest import AbstractPLSTest
from ...test.misc import TestRegression, Tags, Test
from ....algorithm.pls import DIPLS
from ....core.matrix import Matrix
from ....core.matrix.factory import randn_like


class DIPLSTest(AbstractPLSTest[DIPLS]):
    @TestRegression
    def lambda_01(self):
        self.subject.lambda_ = 0.01

    def setup_regressions(self, subject: DIPLS, input_data: List[Matrix]):
        x_source_domain: Matrix = input_data[0]
        y_source_domain: Matrix = input_data[1]
        x_target_domain: Matrix = x_source_domain.add(randn_like(x_source_domain, 0, 1, 2))
        y_target_domain: Matrix = y_source_domain.add(randn_like(y_source_domain, 1, 1, 2))
        x_target_domain_unlabeled: Matrix = x_source_domain.add(randn_like(x_source_domain, 100, 1, 2))

        # Initialise supervised
        subject.initialize_supervised(x_source_domain, x_target_domain, y_source_domain, y_target_domain)
        self.add_default_pls_matrices(subject, x_target_domain, Tags.SUPERVISED)
        subject.reset()

        # Initialise unsupervised
        subject.initialize_unsupervised(x_source_domain, x_target_domain, y_source_domain)
        self.add_default_pls_matrices(subject, x_target_domain, Tags.UNSUPERVISED)
        subject.reset()

        # Initialise semisupervised
        subject.initialize_semisupervised(x_source_domain, x_target_domain, x_target_domain_unlabeled,
                                          y_source_domain, y_target_domain)
        self.add_default_pls_matrices(subject, x_target_domain, Tags.SEMISUPERVISED)

    @Test
    def check_transformed_num_components(self):
        X: Matrix = self.input_data[0]
        X_2: Matrix = X.add(randn_like(X, 0, 0, 2))
        Y: Matrix = self.input_data[1]

        for i in range(1, 5):
            self.subject.num_components = i
            self.subject.initialize_unsupervised(X, X_2, Y)
            transform: Matrix = self.subject.transform(X)
            self.assertEqual(i, transform.num_columns())

            # Reset
            self.subject = self.instantiate_subject()

    def instantiate_subject(self) -> DIPLS:
        return DIPLS()
