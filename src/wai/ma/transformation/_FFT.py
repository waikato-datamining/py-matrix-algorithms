#  _FFT.py
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
from enum import Enum

import numpy as np
from wai.common import switch, case, default

from ..core.matrix import Matrix
from ._AbstractTransformation import AbstractTransformation


class OutputMode(Enum):
    """
    Enumeration of the different types of output the FFT transformation
    can produce.
    """
    REAL_COMPONENT_ONLY = 1
    IMAG_COMPONENT_ONLY = 2
    REAL_AND_IMAG_PAIRS = 3
    AMPLITUDE_ONLY = 4
    PHASE_ANGLE_ONLY = 8
    AMPL_ANGLE_PAIRS = 12


class FFT(AbstractTransformation):
    def __init__(self):
        super().__init__()

        # Options
        self.orthonormalisation: bool = False
        self.output_mode: OutputMode = OutputMode.AMPLITUDE_ONLY

    def __setattr__(self, key, value):
        # Can set output mode by int as well
        if key == "output_mode" and isinstance(value, int):
            value = OutputMode(value)

        # Otherwise just as normal
        super().__setattr__(key, value)

    def configure(self, data: Matrix):
        # No config needed
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        samples: np.ndarray = data.data

        fourier: np.ndarray = self.fft(samples)

        internal_formatted: np.ndarray = self.internal_format(fourier)

        output_formatted: np.ndarray = self.format_output(internal_formatted)

        return Matrix(output_formatted)

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        output_formatted: np.ndarray = data.data

        internal_formatted: np.ndarray = self.inverse_format_output(output_formatted)

        fourier: np.ndarray = self.internal_unformat(internal_formatted)

        samples: np.ndarray = self.ifft(fourier)

        return Matrix(samples)

    def fft(self, array: np.ndarray) -> np.ndarray:
        """
        Gets the FFT of the given array.

        :param array:           The array to FFT.
        :return:                The FFT of the array.
        """
        return np.fft.fftshift(
            np.fft.fft(
                array,
                axis=1,
                norm="ortho" if self.orthonormalisation else None
            )
        )

    def ifft(self, array: np.ndarray) -> np.ndarray:
        """
        Gets the inverse FFT of the given array.

        :param array:           The array to inverse FFT.
        :return:                The inverse FFT of the array.
        """
        return np.fft.ifft(
            np.fft.ifftshift(array),
            axis=1,
            norm="ortho" if self.orthonormalisation else None
        )

    @staticmethod
    def internal_format(fourier: np.ndarray) -> np.ndarray:
        """
        Formats the Fourier transform (complex) array into a real
        array consisting of the real components, optionally paired with
        the necessary imaginary components of the complex numbers.

        :param fourier:     The result of the FFT.
        :return:            The internally-formatted array.
        """
        # Get the number of columns in the array
        num_columns: int = fourier.shape[1]

        # Whether we need the imaginary component of the Nyquist frequency
        need_nyquist: bool = num_columns % 2 == 0

        # Initialise the formatted columns with the real part of the zero-frequency term
        columns = [np.real(fourier[:, num_columns // 2])]

        # Add the real and imaginary parts of the positive frequencies
        for i in range(num_columns // 2 + 1, num_columns):
            column = fourier[:, i]
            columns.append(np.real(column))
            columns.append(np.imag(column))

        # Add the real part of the Nyquist frequency if needed
        if need_nyquist:
            columns.append(np.real(fourier[:, 0]))

        # Return the stacked columns
        return np.stack(columns, axis=1)

    @staticmethod
    def internal_unformat(internal_formatted: np.ndarray) -> np.ndarray:
        """
        Inverse of internal format. Restores FFT result from internally-formatted
        array.

        :param internal_formatted:  The internally-formatted array.
        :return:                    The FFT.
        """
        # Get the number of columns
        num_columns: int = internal_formatted.shape[1]

        # Work out if we have a Nyquist frequency component
        has_nyquist: bool = num_columns % 2 == 0

        # Create lists of the real and imaginary columns,
        # initialised with the zero-frequency term
        real_columns = [internal_formatted[:, 0]]
        imag_columns = [np.zeros_like(real_columns[0])]

        # Add the formatted columns to their respective lists
        for i in range(1, num_columns):
            respective_list = real_columns if i % 2 == 1 else imag_columns
            respective_list.append(internal_formatted[:, i])

        # Add the (zero) imaginary part of the Nyquist term,
        # if it's present
        if has_nyquist:
            imag_columns.append(np.zeros_like(real_columns[0]))

        # Recreate the negative frequencies
        real_columns = [column for column in reversed(real_columns[1:])] + real_columns
        imag_columns = [-1 * column for column in reversed(imag_columns[1:])] + imag_columns

        # If we has a Nyquist frequency, we now have two,
        # so discard the end one
        if has_nyquist:
            real_columns = real_columns[:-1]
            imag_columns = imag_columns[:-1]

        # Recompose the real and imaginary parts
        real = np.stack(real_columns, axis=1)
        imag = 1j * np.stack(imag_columns, axis=1)

        # Return the complex sum
        return real + imag

    def format_output(self, array: np.ndarray) -> np.ndarray:
        """
        Formats the (complex) output array based on the given mode. Inverse
        function of inverse_format_output.

        :param array:   The complex array.
        :return:        The formatted array
        """
        with switch(self.output_mode):
            if case(OutputMode.REAL_COMPONENT_ONLY):
                return format_real_component_only(array)
            if case(OutputMode.IMAG_COMPONENT_ONLY):
                return format_imag_component_only(array)
            if case(OutputMode.REAL_AND_IMAG_PAIRS):
                return format_real_and_imag_pairs(array)
            if case(OutputMode.AMPLITUDE_ONLY):
                return format_amplitude_only(array)
            if case(OutputMode.PHASE_ANGLE_ONLY):
                return format_phase_angle_only(array)
            if case(OutputMode.AMPL_ANGLE_PAIRS):
                return format_ampl_angle_pairs(array)

    def inverse_format_output(self, array: np.ndarray) -> np.ndarray:
        """
        Removes the formatting on the given array for the given output mode.
        Inverse function of format_output.

        :param array:   The formatted array.
        :return:        The unformatted array.
        """
        with switch(self.output_mode):
            if case(OutputMode.REAL_AND_IMAG_PAIRS):
                return inverse_format_real_and_imag_pairs(array)
            if case(OutputMode.AMPL_ANGLE_PAIRS):
                return inverse_format_ampl_angle_pairs(array)
            if default():
                raise ValueError("Inverse transform only available for modes pair modes, not " + self.output_mode.name)


def format_real_component_only(array: np.ndarray) -> np.ndarray:
    columns = [array[:, 0]]

    for i in range(1, array.shape[1], 2):
        columns.append(array[:, i])

    return np.stack(columns, axis=1)


def format_imag_component_only(array: np.ndarray) -> np.ndarray:
    columns = [np.zeros_like(array[:, 0])]

    for i in range(2, array.shape[1], 2):
        columns.append(array[:, i])

    if array.shape[1] % 2 == 0:
        columns.append(np.zeros_like(array[:, 0]))

    return np.stack(columns, axis=1)


def format_real_and_imag_pairs(array: np.ndarray) -> np.ndarray:
    return array


def format_amplitude_only(array: np.ndarray) -> np.ndarray:
    columns = [array[:, 0] + 0j]

    for i in range(2, array.shape[1], 2):
        columns.append(array[:, i - 1] + 1j * array[:, i])

    if array.shape[1] % 2 == 0:
        columns.append(array[:, -1] + 0j)

    return np.abs(np.stack(columns, axis=1))


def format_phase_angle_only(array: np.ndarray) -> np.ndarray:
    columns = [array[:, 0] + 0j]

    for i in range(2, array.shape[1], 2):
        columns.append(array[:, i - 1] + 1j * array[:, i])

    if array.shape[1] % 2 == 0:
        columns.append(array[:, -1] + 0j)

    return np.angle(np.stack(columns, axis=1))


def format_ampl_angle_pairs(array: np.ndarray) -> np.ndarray:
    columns = [array[:, 0]]

    for i in range(2, array.shape[1], 2):
        compl = array[:, i - 1] + 1j * array[:, i]
        columns.append(np.abs(compl))
        columns.append(np.angle(compl))

    if array.shape[1] % 2 == 0:
        columns.append(array[:, -1])

    return np.stack(columns, axis=1)


def inverse_format_real_and_imag_pairs(array: np.ndarray) -> np.ndarray:
    return array


def inverse_format_ampl_angle_pairs(array: np.ndarray) -> np.ndarray:
    columns = [array[:, 0]]

    for i in range(2, array.shape[1], 2):
        ampl = array[:, i - 1]
        angle = array[:, i]
        compl = ampl * (np.cos(angle) + 1j * np.sin(angle))
        columns.append(np.real(compl))
        columns.append(np.imag(compl))

    if array.shape[1] % 2 == 0:
        columns.append(array[:, -1])

    return np.stack(columns, axis=1)
