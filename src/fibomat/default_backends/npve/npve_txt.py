"""Provides the :class:`NPVETxt` backend, which saves a deflection list for NPVE.

Example::

    exported = layout.export(NPVETxt)
    exported.save('points.txt')
"""
from __future__ import annotations

import typing as t
import warnings

import numpy as np

from fibomat import utils
from fibomat.default_backends.spotlist_backend import SpotListBackend
from fibomat.units import unit


__all__ = ['NPVETxt']


def _save_impl(filename: utils.PathLike, dwell_points: np.ndarray, parameters: t.Dict[str, t.Any]) -> None:
    """Save the dwell points as NPVE deflection list (the dwell time is a multiple of 0.1 µs)."""
    if "base_dwell_time" not in parameters or not parameters["base_dwell_time"]:
        raise RuntimeError('The base dwell time must be set.')

    warnings.warn(
        "The base dwell time is set to 0.1 us by default. Open a bug report if you need a different base dwell time.",
        stacklevel=2
    )

    fov = max(parameters["fov"].width, parameters["fov"].height)

    # newline='': the lines end with "\r\n" on all platforms
    with open(filename, "w", newline="", encoding="utf-8") as file:
        file.write("NPVE DEFLECTION LIST\r\n")
        file.write("UNITS=MICRONS\r\n")
        file.write("DWELL=0.1\r\n")
        file.write(f"FOV={fov}\r\n")
        file.write("START\r\n")
        np.savetxt(file, dwell_points, "%.5f %.5f %d", newline="\r\n")
        file.write("END\r\n")


class NPVETxt(SpotListBackend):
    """Rasterizes all patterns and saves the dwell points as NPVE deflection list: positions in µm and dwell times as
    multiples of 0.1 µs. The field of view is the extent of the points (see :class:`SpotListBackend`)."""

    def __init__(self, description: t.Optional[str] = None):
        """
        Args:
            description (str, optional): description
        """
        super().__init__(
            save_impl=_save_impl,
            base_dwell_time=0.1 * unit('µs'),
            length_unit=unit('µm'),
            time_unit=unit('µs'),
            description=description,
        )
