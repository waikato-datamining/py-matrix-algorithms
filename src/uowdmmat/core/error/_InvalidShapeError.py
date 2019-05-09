class InvalidShapeError(RuntimeError):
    def __init__(self, message: str, *matrix_shapes: str):
        if matrix_shapes is None or len(matrix_shapes) == 0:
            super().__init__('Invalid shape ' + message)
        else:
            super().__init__('Invalid shapes '
                             + ', '.join([m for m in matrix_shapes])
                             + ' '
                             + message
                             )
