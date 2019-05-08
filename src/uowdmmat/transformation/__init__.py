from uowdmmat.core.matrix import Matrix

# Aliases
from uowdmmat.transformation.Center import Center
from uowdmmat.transformation.RowNorm import RowNorm
from uowdmmat.transformation.Standardize import Standardize


def center(data: Matrix) -> Matrix:
    return Center().transform(data)


def row_norm(data: Matrix) -> Matrix:
    return RowNorm().transform(data)


def standardize(data: Matrix) -> Matrix:
    return Standardize().transform(data)

