from typing import Optional

from ..core import real
from ..core.matrix import Matrix
from ..core.matrix.helper import row_means, row_stdevs
from ._AbstractTransformation import AbstractTransformation


class RowNorm(AbstractTransformation):
    def __init__(self):
        super().__init__()
        self.means: Optional[Matrix] = None
        self.std_devs: Optional[Matrix] = None

    def reset(self):
        super().reset()
        self.means = None
        self.std_devs = None

    def configure(self, data: Matrix):
        self.means = row_means(data)
        self.std_devs = row_stdevs(data)
        # Make sure we don't do a divide by zero
        for i in range(self.std_devs.num_rows()):
            if self.std_devs.get(i, 0) == real(0):
                self.std_devs.set(i, 0, 1)
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        result = data.copy()
        result.sub_by_vector_modify(self.means)
        result.div_by_vector_modify(self.std_devs)
        return result

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        result = data.copy()
        result.mul_by_vector_modify(self.std_devs)
        result.add_by_vector_modify(self.means)
        return result

    def get_means(self) -> Matrix:
        return self.means

    def get_std_devs(self) -> Matrix:
        return self.std_devs
