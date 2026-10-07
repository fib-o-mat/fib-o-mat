"""Backend for FEI stream files.

Transferred from the old gitlab repo (https://gitlab.com/viggge/fib-o-mat/-/merge_requests/1); code originally
provided by Markus Lid.

Example::

    exported = layout.export(FEIStreamFile, n_rep=3)
    exported.save('filename')  # writes the stream file 'filename(HFW=12µm).str'
"""
from __future__ import annotations

import math
import pathlib
import typing as t
import warnings

import numpy as np

from fibomat import utils
from fibomat.default_backends.spotlist_backend import SpotListBackend
from fibomat.units import scale_factor, unit


__all__ = ['FEIStreamFile', 'stream_file_impl']


def stream_file_impl(
    n_rep: int = 1, margin: float = 0.9, dac16bit: bool = True, hfw_rounding: bool = True,
    time_low_res: t.Union[bool, str] = True
) -> t.Callable[[utils.PathLike, np.ndarray, t.Dict[str, t.Any]], None]:
    """Create the function which saves the dwell points of a :class:`~fibomat.default_backends.SpotListBackend` as
    FEI stream file. The exported file has a filename which includes the horizontal field width (HFW), which has to be
    set at the instrument, e.g. 'filename(HFW=12µm).str'.

    The spot list backend must be used with the length unit µm and a base dwell time (the stream file contains the
    dwell times as multiples of the time resolution of the stream file).

    Example::

        exported = layout.export(
            SpotListBackend, base_dwell_time=0.1 * unit('µs'), length_unit=unit('µm'),
            save_impl=stream_file_impl(n_rep=3)
        )
        exported.save('filename')  # writes a stream file named 'filename(HFW=12µm).str'

    Args:
        n_rep (int): the number of times the whole pattern is repeated. More iterations do not increase the file size.
        margin (float): how much of the imaging area is used for patterning, in [0.1, 1.0].
        dac16bit (bool): if True, a 16 bit digital to analog converter is assumed, otherwise a 12 bit converter. Check
            the manual of your instrument to find out which one is appropriate.
        hfw_rounding (bool): if True, the horizontal field width (HFW) is rounded (see :func:`round_hfw` in the
            source) to a value the instrument can set.
        time_low_res (bool, str): the time resolution of the stream file with a 16 bit converter. Pass ``'low'``
            for 100 ns; anything else (also the default True) gives 25 ns. A 12 bit converter always has 100 ns.

    Returns:
        Callable: save function for the :class:`~fibomat.default_backends.SpotListBackend`

    Raises:
        ValueError: Raised if n_rep is smaller than 1 or margin is not in [0.1, 1].
    """
    if int(n_rep) != n_rep or n_rep < 1:
        raise ValueError(f'n_rep must be an integer which is not smaller than 1, got {n_rep}.')
    if not 0.1 <= margin <= 1.:
        raise ValueError(f'margin must be in [0.1, 1], got {margin}.')

    def streamfile_type() -> t.Tuple[int, int, int, str]:
        """Resolutions of the positions (x, y) and the time (ns) and the header line of the stream file."""
        if dac16bit:
            x_res = 65536  # 2^16 = 65536
            # actually the user manual is lying, y resolution is the same you just can't see it on the screen anymore
            y_res = x_res  # 56576
            if time_low_res == 'low':
                time_unit = 100  # [ns]
                header = 's16\n'
            else:
                time_unit = 25  # [ns]
                header = 's16,25ns\n'
        else:
            x_res = 4095  # 2^12 = 4095
            # resolution is smaller in y direction. Ref. user manual.
            y_res = 3816
            header = 's\n'
            time_unit = 100  # [ns]
            if not time_low_res:
                warnings.warn("High resolution is not allowed with 12 bit DAC. Using 'low' instead", stacklevel=3)

        return x_res, y_res, time_unit, header

    def round_hfw(hfw: float) -> float:
        if hfw_rounding:
            # (rounding to the next multiple of a quarter of the order of magnitude and then to an integer, in µm)
            oom = math.floor(math.log(hfw, 10))  # order of magnitude
            hfw = np.round(np.ceil(hfw * 4 * 10 ** (-oom)) / (4 * 10 ** (-oom)))
        return hfw

    def save_impl(filename: utils.PathLike, dwell_points: np.ndarray, parameters: t.Dict[str, t.Any]) -> None:
        # fov is in units of length_unit (µm)
        x_res, y_res, time_unit, header = streamfile_type()

        base_dwell_time = parameters["base_dwell_time"]
        if base_dwell_time is None:
            raise ValueError('The spot list backend needs a base dwell time for stream files.')

        fov = parameters["fov"]
        center = np.asarray(fov.center)
        width, height = fov.width, fov.height

        xy_aspect_ratio = x_res / y_res
        # Setting the horizontal field width (HFW) based on which is FOV aspect ratio
        if height != 0 and width / height > xy_aspect_ratio:  # (single points have no height)
            hfw = width / margin
        else:
            hfw = height * xy_aspect_ratio / margin
        if hfw > 0:
            hfw = round_hfw(hfw)
        if hfw == 0:
            warnings.warn("calculated horizontal field width to be 0. hfw was set to 1", stacklevel=2)
            hfw = 1

        shift = center - np.array([hfw / 2, hfw / xy_aspect_ratio / 2])
        dwell_points[:, 0:2] = (dwell_points[:, 0:2] - shift) * x_res / hfw

        # The dwell times are multiples of the base dwell time and are converted to multiples of the time resolution.
        # (`scale_factor` ignores the magnitude of the base dwell time, so it is multiplied afterwards.)
        dwell_points[:, 2] *= (scale_factor(unit('ns'), base_dwell_time) * base_dwell_time.magnitude) / time_unit
        dwell_points = dwell_points.round()

        path = pathlib.Path(filename)
        path = path.with_name(path.name.split('.', 1)[0] + f'(HFW={hfw:0.0f}{parameters["length_unit"]:~P})' + '.str')

        with open(path, 'w', encoding='utf-8') as file:
            # first, write header data.
            file.writelines([
                header,
                f'{n_rep:d}\n',  # number of times to repeat the pattern
                f'{parameters["number_of_points"]:d}\n',
            ])

            # second, write dwell point data
            # (x, y, t_d) where x and y are the position of a spot and t_d the dwell time or dwell time multiplicand.
            # FEI stream files take the dwell time column first, therefore it is reordered. All values in stream
            # files are given as integers.
            np.savetxt(file, dwell_points[:, [2, 0, 1]], "%d %d %d")

    return save_impl


class FEIStreamFile(SpotListBackend):
    """Rasterizes all patterns and saves them as FEI stream file (see :func:`stream_file_impl`). The base dwell time is
    0.1 µs and the positions are in µm. The saved file name contains the horizontal field width (HFW)."""

    def __init__(
        self,
        n_rep: int = 1,
        margin: float = 0.9,
        dac16bit: bool = True,
        hfw_rounding: bool = True,
        time_low_res: t.Union[bool, str] = True,
        description: t.Optional[str] = None
    ):
        """
        See :func:`stream_file_impl` for the description of the arguments.

        Args:
            n_rep (int): the number of times the whole pattern is repeated
            margin (float): how much of the imaging area is used for patterning, in [0.1, 1.0]
            dac16bit (bool): if True, a 16 bit digital to analog converter is assumed, otherwise a 12 bit converter
            hfw_rounding (bool): if True, the horizontal field width is rounded
            time_low_res (bool, str): time resolution of the stream file
            description (str, optional): description
        """
        super().__init__(
            save_impl=stream_file_impl(n_rep, margin, dac16bit, hfw_rounding, time_low_res),
            base_dwell_time=0.1 * unit('µs'),
            length_unit=unit('µm'),
            time_unit=unit('µs'),
            description=description,
        )
