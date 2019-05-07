from typing import Optional

from core.matrix import Matrix
from core.matrix.helper import column_means
from transformation.AbstractTransformation import AbstractTransformation


class Center(AbstractTransformation):
    def __init__(self):
        super().__init__()
        self.means: Optional[Matrix] = None

    def reset(self):
        super().reset()
        self.means = None

    def configure(self, data: Matrix):
        self.means = column_means(data)
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        result = data.copy()
        result.sub_by_vector_modify(self.means)
        return result

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        result = data.copy()
        result.add_by_vector_modify(self.means)
        return result
