"""Provide the :class:`DDDMill` class.

.. todo:: Not reworked yet (see ``TODO.md`` in the repository root).
"""
import types

import pint

from fibomat.mill.mill_base import MillBase


__all__ = ['DDDMill']


class DDDMill(MillBase):
    """The `DDDMill` class is used to specify the dwell_time per spot and the number of repeats for a pattern.

    Optionally, the class can hold an object describing the shape of the ion beam which is needed if any kind of
    optimization is done.
    """
    def __init__(self, dwell_time: types.FunctionType, repeats: int):

            try:
                if dwell_time.__annotations__["return"] is not pint.registry.Quantity:
                    raise TypeError("dwell_time must give quantities")
            except:  # User should be free not to typehint
                print("No Typehint for dwell_time, please check if function returns quantites.")
            if not isinstance(repeats, int):
                raise TypeError('repats must be an int')

            if repeats < 1:
                raise ValueError('repeats must be at least 1.')

            super().__init__(dwell_time=dwell_time, repeats=repeats)
    @property
    def dwell_time(self) -> types.FunctionType:
        return self['dwell_time']

    @property
    def repeats(self) -> int:
        return self['repeats']
