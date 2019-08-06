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
import numpy as np

from ..test.misc import TestRegression, Test
from ..transformation import AbstractTransformationTest
from wai.ma.transformation import FFT, OutputMode
from wai.ma.core.matrix import Matrix


class FFTTest(AbstractTransformationTest[FFT]):
    @TestRegression
    def orthonormal(self):
        self.subject.orthonormalisation = True

    @TestRegression
    def real_imag_pairs(self):
        self.subject.output_mode = OutputMode.REAL_AND_IMAG_PAIRS

    @Test
    def inverses(self):
        input: Matrix = self.input_data[0]

        # FFT and inverse
        fft: np.ndarray = self.subject.fft(input.data)
        ifft: np.ndarray = self.subject.ifft(fft)

        # Check if ifft == input.data
        is_equal: bool = np.allclose(ifft, input.data)
        self.assertTrue(is_equal, "Full-circle inversion of FFT failed")

        # Internal format and inverse
        internal_format: np.ndarray = self.subject.internal_format(fft)
        internal_unformat: np.ndarray = self.subject.internal_unformat(internal_format)

        # Check if internal_unformat == fft
        is_equal: bool = np.allclose(internal_unformat, fft)
        self.assertTrue(is_equal, "Full-circle inversion of internal format failed")

        # Output format and inverse
        output_format: np.ndarray = self.subject.format_output(internal_format)
        output_unformat: np.ndarray = self.subject.inverse_format_output(output_format)

        # Check if output_unformat == internal_format
        is_equal: bool = np.allclose(output_unformat, internal_format)
        self.assertTrue(is_equal, "Full-circle inversion of output format failed")

    def instantiate_subject(self) -> FFT:
        fft = FFT()
        fft.output_mode = OutputMode.AMPL_ANGLE_PAIRS
        return fft
