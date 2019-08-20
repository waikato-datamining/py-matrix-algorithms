#  _EquidistanceFilter.py
#  Copyright (C) 2019 University of Waikato, Hamilton, New Zealand
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
from ..core import Filter
from ..core.matrix import Matrix, factory


class Equidistance(Filter):
    """
    Filter which resamples rows to the given number of columns
    using linear interpolation.
    """
    def __init__(self, num_samples: int):
        self.num_samples: int = num_samples  # The number of samples to resample to

    def __setattr__(self, key, value):
        # Must be at least 2 samples in the output
        if key == "num_samples" and value < 2:
            raise ValueError("num_samples must be at least 2")

        super().__setattr__(key, value)

    def transform(self, predictors: Matrix) -> Matrix:
        # Create a 1-dimensional space where the sample positions
        # for the original and transformed matrices are integral
        resampled_step = (predictors.num_columns() - 1)
        original_step = (self.num_samples - 1)

        # Create a correctly-sized matrix to hold the resampled columns
        resampled = factory.zeros(predictors.num_rows(), self.num_samples)

        # Calculate each resampled column in turn
        for resampled_index in range(self.num_samples):
            # Get the x-position of this sample in our integral space
            x = resampled_step * resampled_index

            # Get the index of the original column at (or just
            # to the left of) this x-position
            old_index = x // original_step

            # If there is an original column at exactly this position,
            # just copy it
            if x % original_step == 0:
                resampled.set_column(resampled_index, predictors.get_column(old_index))
                continue

            # Get the columns either side of the x-position
            left = predictors.get_column(old_index)
            right = predictors.get_column(old_index + 1)

            # Get the interpolation factor for the x-position
            t = (x % original_step) / original_step

            # Add the linear interpolation of the 2 columns
            resampled.set_column(resampled_index, left.mul(1 - t).add(right.mul(t)))

        return resampled

