from ._SavitzkyGolay import SavitzkyGolay


class SavitzkyGolay2(SavitzkyGolay):
    def __init__(self):
        super().__init__()
        self.num_points: int = 3

        # Delete num_points_left/right
        del self.num_points_left
        del self.num_points_right

    def __getattribute__(self, item):
        # Alias num_points_left/right to num_points
        if item in {'num_points_left', 'num_points_right'}:
            return self.num_points

        return super().__getattribute__(item)

    @staticmethod
    def validate_num_points(value: int):
        if value < 0:
            raise ValueError('num_points must be at least 0')
