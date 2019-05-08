# Aliases
from ._Center import Center
from ._RowNorm import RowNorm
from ._Standardize import Standardize
from ._SavitzkyGolay import SavitzkyGolay
from ._SavitzkyGolay2 import SavitzkyGolay2
from ._MultiplicativeScatterCorrection import MultiplicativeScatterCorrection

center = Center.quick_apply
row_norm = RowNorm.quick_apply
standardize = Standardize.quick_apply
