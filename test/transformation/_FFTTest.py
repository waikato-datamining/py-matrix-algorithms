#  _FFTTest.py
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
from ..test.misc import TestRegression
from ..transformation import AbstractTransformationTest
from wai.ma.transformation import FFT, OutputMode


class FFTTest(AbstractTransformationTest[FFT]):
    @TestRegression
    def orthonormal(self):
        self.subject.orthonormalisation = True

    @TestRegression
    def real_imag_pairs(self):
        self.subject.output_mode = OutputMode.REAL_AND_IMAG_PAIRS

    def instantiate_subject(self) -> FFT:
        fft = FFT()
        fft.output_mode = OutputMode.AMPL_ANGLE_PAIRS
        return fft
