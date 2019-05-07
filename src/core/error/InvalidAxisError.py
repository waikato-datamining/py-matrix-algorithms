class InvalidAxisError(RuntimeError):
    def __init__(self, axis: int):
        super().__init__('Axis has to be either 0 or 1 but was ' + str(axis) + '.')