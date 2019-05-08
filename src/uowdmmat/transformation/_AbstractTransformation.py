from abc import abstractmethod, ABC

from ..core.matrix import Matrix


class AbstractTransformation(ABC):
    def __init__(self):
        self.configured: bool = False
        pass

    def reset(self):
        self.configured = False

    @abstractmethod
    def configure(self, data: Matrix):
        pass

    @abstractmethod
    def do_transform(self, data: Matrix) -> Matrix:
        pass

    def transform(self, data: Matrix) -> Matrix:
        if not self.configured:
            self.configure(data)
        return self.do_transform(data)

    @abstractmethod
    def do_inverse_transform(self, data: Matrix) -> Matrix:
        pass

    def inverse_transform(self, data: Matrix) -> Matrix:
        if not self.configured:
            self.configure(data)
        return self.do_inverse_transform(data)

    @classmethod
    def quick_apply(cls, data: Matrix) -> Matrix:
        return cls().transform(data)

