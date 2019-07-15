#  _Tags.py
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

"""
Tags that should be used when calling AbstractRegressionTest.add_regression(str, Union[Matrix, real]).
"""

TRANSFORM = 'transform'
INVERSE_TRANSFORM = 'inverse-transform'
LOADINGS = 'loadings'
SCORES = 'scores'
PREDICTIONS = 'predictions'
PROJECTION = 'projection'
MATRIX = 'matrix'
SUPERVISED = 'supervised'
SEMISUPERVISED = 'semisupervised'
UNSUPERVISED = 'unsupervised'
