from typing import Optional

from uowdmmat.core.matrix import Matrix, real
from uowdmmat.core.matrix.helper import column_stdevs, column_means
from uowdmmat.transformation.AbstractTransformation import AbstractTransformation


class Standardize(AbstractTransformation):
    def __init__(self):
        super().__init__()
        self.means: Optional[Matrix] = None
        self.std_devs: Optional[Matrix] = None

    def reset(self):
        super().reset()
        self.means = None
        self.std_devs = None

    def configure(self, data: Matrix):
        self.means = column_means(data)
        self.std_devs = column_stdevs(data)
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
