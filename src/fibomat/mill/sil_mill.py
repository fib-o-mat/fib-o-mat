"""Provide the :class:`SILMill` class.

.. todo:: Not reworked yet (see ``TODO.md`` in the repository root).
"""
import numpy as np

from fibomat.mill.ddd_mill import DDDMill
from fibomat.units import QuantityType, has_length_dim, Q_, scale_to


__all__ = ['SILMill']


class SILMill(DDDMill):
    def __init__(self, max_dwell_time: QuantityType, radius_sil: QuantityType, radius: QuantityType, repeats: int, min_dwell_time=1):
        if not isinstance(repeats, int):
            raise TypeError('repeats must be an int')
        if repeats < 1:
            raise ValueError('repeats must be at least 1.')
        if not isinstance(radius_sil, Q_) or not isinstance(radius, Q_):
            raise TypeError('both radii must be a quantity.')

        if not has_length_dim(radius_sil) or not has_length_dim(radius):
            raise ValueError('both radii must have dimension [length].')

        # Store as quantities (with units if possible)
        self._radius_sil = radius_sil.magnitude
        self._radius = scale_to(radius_sil, radius)

        self._radii_unit = radius_sil.units  # will be set by set_unit

        def dwell_func(point: np.ndarray) -> QuantityType:
            x, y = point[0], point[1]
            dist_sq = x * x + y * y
            dist = np.sqrt(dist_sq)
            # Use the already scaled radii
            radius_sil = self._radius_sil
            radius = self._radius
            if dist < radius_sil:
                depth = max_dwell_time - (max_dwell_time / radius_sil) * np.sqrt(radius_sil ** 2 - dist_sq)
            elif dist < radius:
                depth = max_dwell_time * (1 - (dist - radius_sil) / (radius - radius_sil))
            else:
                return Q_(0, 'microsecond')
            return Q_(max(depth, min_dwell_time), 'microsecond')

        super().__init__(dwell_time=dwell_func, repeats=repeats)

    def set_unit(self, length_unit):
        """Scale radii to the given unit (e.g. 'nm', 'µm'). Call this ONCE before rasterization."""
        self._radius_sil = scale_to(length_unit, self._radii_unit*self._radius_sil)
        self._radius = scale_to(length_unit, self._radii_unit*self._radius)
        self._radii_unit = length_unit
