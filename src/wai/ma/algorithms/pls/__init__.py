#  __init__.py
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

from ._AbstractMultiResponsePLS import AbstractMultiResponsePLS
from ._AbstractPLS import AbstractPLS
from ._AbstractSingleResponsePLS import AbstractSingleResponsePLS
from ._CCARegression import CCARegression
from ._DIPLS import DIPLS
from ._KernelPLS import KernelPLS
from ._NIPALS import NIPALS, DeflationMode, WeightCalculationMode
from ._OPLS import OPLS
from ._PLS1 import PLS1
from ._PreprocessingType import PreprocessingType
from ._PRM import PRM
from ._SIMPLS import SIMPLS
from ._SparsePLS import SparsePLS
from ._VCPLS import VCPLS
