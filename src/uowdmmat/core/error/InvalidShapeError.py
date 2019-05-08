from uowdmmat.core.matrix import Matrix


class InvalidShapeError(RuntimeError):
    def __init__(self, message: str, *matrices: Matrix):
        if matrices is None or len(matrices) == 0:
            super().__init__('Invalid shape ' + message)
        else:
            super().__init__('Invalid shapes '
                             + ', '.join([m.shape_string() for m in matrices])
                             + ' '
                             + message
                             )