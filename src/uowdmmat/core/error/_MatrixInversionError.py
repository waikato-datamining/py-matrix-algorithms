class MatrixInversionError(RuntimeError):

    prefix: str = 'Could not invert matrix. '

    def __init__(self, message: str):
        super().__init__(MatrixInversionError.prefix + message)
