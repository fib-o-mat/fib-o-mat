"""Provide the :class:`SpecialMill` class."""
from fibomat.mill.mill_base import MillBase


__all__ = ['SpecialMill']


class SpecialMill(MillBase):
    """Mill with arbitrary, backend specific settings.

    Example:
        >>> mill = SpecialMill(beam_current=1., aperture=3)
        >>> mill['aperture']
        3
    """
