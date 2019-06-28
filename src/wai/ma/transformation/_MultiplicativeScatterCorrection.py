#  _MultiplicativeScatterCorrection.py
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

import builtins
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

from ..core import utils, real
from ..core.matrix import Matrix, helper
from ._AbstractTransformation import AbstractTransformation


class MultiplicativeScatterCorrection(AbstractTransformation):
    def __init__(self):
        super().__init__()
        self.prefilter: Optional[AbstractTransformation] = None
        self.correction: AbstractMultiplicativeScatterCorrection = RangeBased()
        self.average: Optional[Matrix] = None

    @staticmethod
    def create_configuration_matrix(spectra: List[Matrix]) -> Matrix:
        if len(spectra) == 0:
            raise ValueError("Can't create configuration matrix for 0 spectra")

        wave_numbers = spectra[0].get_column(0)
        amplitudes = [spectrum.get_column(1) for spectrum in spectra]

        return helper.multi_concat(1, wave_numbers, *amplitudes)

    def configure(self, data: Matrix):
        wave_numbers = data.get_column(0)
        amplitudes = data.get_sub_matrix((0, data.num_rows()), (1, data.num_columns()))

        self.average = wave_numbers.concat_along_columns(amplitudes.mean(1))
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        return self.correction.correct(self.average, data)

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        raise NotImplementedError('Cannot perform inverse transform of Multiplicative Scatter Correction')


class AbstractMultiplicativeScatterCorrection(ABC):
    def __init__(self):
        self.prefilter: Optional[AbstractTransformation] = None

    @abstractmethod
    def correct(self, average: Matrix, data: Matrix) -> Matrix:
        pass


class RangeBased(AbstractMultiplicativeScatterCorrection):
    def __init__(self):
        super().__init__()

        self.ranges: List[Tuple[real, real]] = []

    def correct(self, average: Matrix, data: Matrix) -> Matrix:
        result = data.copy()

        if self.prefilter is not None:
            filtered = self.prefilter.transform(data)
        else:
            filtered = data

        for range in self.ranges:
            x = []
            y = []
            wave = []
            for i in builtins.range(average.num_rows()):
                if self.range_contains(range, filtered.get(i, 0)):
                    wave.append(filtered.get(i, 0))
                    y.append(filtered.get(i, 1))
                    x.append(average.get(i, 1))

            inter, slope = utils.linear_regression(x, y)

            for i in builtins.range(result.num_rows()):
                if self.range_contains(range, result.get(i, 0)):
                    result.set(i, 1, (result.get(i, 1) - inter) / slope)

        return result

    @staticmethod
    def range_contains(range: Tuple[real, real], value: real) -> bool:
        lower, upper = range

        return lower <= value <= upper
