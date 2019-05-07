from core.matrix import Matrix

# Aliases
from transformation.Center import Center
from transformation.RowNorm import RowNorm
from transformation.Standardize import Standardize
from transformation.SavitzkyGolayFilter import SavitzkyGolayFilter


def center(data: Matrix) -> Matrix:
    return Center().transform(data)


def row_norm(data: Matrix) -> Matrix:
    return RowNorm().transform(data)


def standardize(data: Matrix) -> Matrix:
    return Standardize().transform(data)


def savitzky_golay_filter(data: Matrix) -> Matrix:
    return SavitzkyGolayFilter().transform(data)
