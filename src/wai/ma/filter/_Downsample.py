from ..core import Filter
from ..core.matrix import Matrix


class Downsample(Filter):
    """
    Filter which gets every Nth column from a matrix, starting at a given index.
    """
    def __init__(self):
        self.start_index: int = 0  # The index to start sampling from
        self.step: int = 1  # The step-size between samples

    def transform(self, predictors: Matrix) -> Matrix:
        columns = [i for i in range(self.start_index, predictors.num_columns(), self.step)]

        return predictors.get_columns(columns)
