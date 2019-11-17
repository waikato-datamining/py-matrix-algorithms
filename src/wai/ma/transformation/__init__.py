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

# Aliases
from ._AbstractTransformation import AbstractTransformation
from ._Center import Center
from ._FFT import FFT, OutputMode
from ._Log import Log
from ._PowerTransformer import PowerTransformer
from ._QuantileTransformer import QuantileTransformer
from ._RobustScaler import RobustScaler
from ._RowNorm import RowNorm
from ._Standardize import Standardize
from ._SavitzkyGolay import SavitzkyGolay
from ._SavitzkyGolay2 import SavitzkyGolay2
from ._MultiplicativeScatterCorrection import MultiplicativeScatterCorrection
from ._PassThrough import PassThrough

center = Center.quick_apply
row_norm = RowNorm.quick_apply
standardize = Standardize.quick_apply
log = Log.quick_apply
