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
from typing import Tuple

import numpy as np

from ..core.matrix import Matrix
from ._AbstractTransformation import AbstractTransformation


class FFT(AbstractTransformation):
    def __init__(self):
        super().__init__()

        # Options
        self.orthonormalisation: bool = False
        self.output_mode: OutputMode = OutputMode.AMPLITUDE_ONLY

    def configure(self, data: Matrix):
        # No config needed
        self.configured = True

    def do_transform(self, data: Matrix) -> Matrix:
        return Matrix(
            format_output(
                self.output_mode,
                fft(data.data, self.orthonormalisation)
            )
        )

    def do_inverse_transform(self, data: Matrix) -> Matrix:
        return Matrix(
            ifft(
                inverse_format_output(self.output_mode, data.data),
                self.orthonormalisation
            )
        )


def fft(array: np.ndarray, orthonormalise: bool) -> np.ndarray:
    """
    Gets the FFT of the given array.

    :param array:           The array to FFT.
    :param orthonormalise:  Whether to use orthogonal normalisation.
    :return:                The FFT of the array.
    """
    return np.fft.fftshift(
        np.fft.fft(
            array,
            axis=1,
            norm="ortho" if orthonormalise else None
        )
    )


def ifft(array: np.ndarray, orthonormalise: bool) -> np.ndarray:
    """
    Gets the inverse FFT of the given array.

    :param array:           The array to inverse FFT.
    :param orthonormalise:  Whether to use orthogonal normalisation.
    :return:                The inverse FFT of the array.
    """
    return np.fft.ifft(
        np.fft.ifftshift(array),
        axis=1,
        norm="ortho" if orthonormalise else None
    )


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


def pairwise_concatenate(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Concatenate the columns of two arrays in order of matching pairs.
    Inverse function of pairwise_separate.

    :param array1:  The first array.
    :param array2:  The second array.
    :return:        The concatenated array.
    """
    # Create a list of column slices
    slices = []

    # Add the columns from the source arrays
    for i in range(array1.shape[1]):
        slices.append(array1[:, i])
        slices.append(array2[:, i])

    # Concatenate the columns
    return np.stack(slices, axis=1)


def pairwise_separate(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separates an array into two sub-arrays, one with the even columns
    and one with the odd columns of the source. Inverse function of
    pairwise_concatenate.

    :param array:   The source array to separate.
    :return:        The two sub-arrays.
    """
    # Create lists to hold the column slices
    slices1, slices2 = [], []

    # Add the columns from the source array to either list
    for i in range(0, array.shape[1], 2):
        slices1.append(array[:, i])
        slices2.append(array[:, i + 1])

    # Return the concatenations of the columns
    return np.stack(slices1, axis=1), np.stack(slices2, axis=1)


def format_output(mode: OutputMode, array: np.ndarray) -> np.ndarray:
    """
    Formats the (complex) output array based on the given mode. Inverse
    function of inverse_format_output.

    :param mode:    The format mode to use.
    :param array:   The complex array.
    :return:        The formatted array
    """
    if mode is OutputMode.REAL_COMPONENT_ONLY:
        return np.real(array)
    elif mode is OutputMode.IMAG_COMPONENT_ONLY:
        return np.imag(array)
    elif mode is OutputMode.REAL_AND_IMAG_PAIRS:
        return pairwise_concatenate(np.real(array), np.imag(array))
    elif mode is OutputMode.AMPLITUDE_ONLY:
        return np.abs(array)
    elif mode is OutputMode.PHASE_ANGLE_ONLY:
        return np.angle(array)
    elif mode is OutputMode.AMPL_ANGLE_PAIRS:
        return pairwise_concatenate(np.abs(array), np.angle(array))
    else:
        raise ValueError("Unexpected output mode: " + str(mode))


def inverse_format_output(mode: OutputMode, array: np.ndarray) -> np.ndarray:
    """
    Removes the formatting on the given array for the given output mode.
    Inverse function of format_output.

    :param mode:    The format mode to use.
    :param array:   The formatted array.
    :return:        The unformatted array.
    """
    if mode is OutputMode.REAL_AND_IMAG_PAIRS:
        real, imag = pairwise_separate(array)
        return real + 1j * imag
    elif mode is OutputMode.AMPL_ANGLE_PAIRS:
        ampl, angle = pairwise_separate(array)
        return ampl * (np.cos(angle) + 1j * np.sin(angle))
    elif isinstance(mode, OutputMode):
        raise ValueError("Inverse transform only available for modes pair modes, not " + mode.name)
    else:
        raise ValueError("Unexpected output mode: " + str(mode))
