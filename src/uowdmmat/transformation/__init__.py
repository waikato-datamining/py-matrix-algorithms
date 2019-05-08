# Aliases
from uowdmmat.transformation._Center import Center
from uowdmmat.transformation._RowNorm import RowNorm
from uowdmmat.transformation._Standardize import Standardize
from uowdmmat.transformation._SavitzkyGolay import SavitzkyGolay
from uowdmmat.transformation._SavitzkyGolay2 import SavitzkyGolay2

center = Center.quick_apply
row_norm = RowNorm.quick_apply
standardize = Standardize.quick_apply
