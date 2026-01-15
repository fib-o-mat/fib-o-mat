"""Provide the :class:`Mill` class."""
from typing import Optional
import types
from fibomat.units import QuantityType, has_time_dim, has_length_dim, Q_
from fibomat.units import Q_, scale_to
from fibomat.mill.ionbeam import IonBeam
import pint
import numpy as np

class MillBase:
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return ('{}(' + ', '.join([key + '={}' for key in self._kwargs.keys()]) + ')').format(
            self.__class__.__name__, *self._kwargs.values()
        )

    # def __getattr__(self, item):
    #     return self.__getitem__(item)
    #
    # def __setattr__(self, key, value):
    #     if not key.startswith('_'):
    #         raise TypeError(
    #             "'MillBase' object does not support item assignment (if you are a developer, use variables with "
    #             " _ [underscore] as first character. Then, this check is bypassed)"
    #         )
    #     super().__setattr__(key, value)

    # def __getattribute__(self, item: str):
    #     print('__getattribute__', item)
    #     if not item.startswith('_') and item not in self._kwargs:
    #         raise AttributeError
    #     return object.__getattribute__(self, item)

        #
    # def __setattr__(self, key, value):
    #     raise NotImplementedError

    def __getitem__(self, item: str):
        try:
            value = self._kwargs[item]

            if value is None:
                raise KeyError('value is none')

            return value
        except KeyError as key_error:
            raise KeyError(
                f'Mill object does not have property "{item}". '
                'Maybe you need a SpecialMill with custom properties for the used backend?'
            ) # from key_error

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

class Mill(DDDMill):
    """The `Mill` class is used to specify a constant dwell_time per spot and the number of repats for a pattern. It should be used for 2D-Shapes.

    Optionally, the class can hold an object describing the shape of the ion beam which is needed if any kind of
    optimization is done.
    """
    def __init__(self, dwell_time: QuantityType, repeats: int):
        """
        Args:
            dwell_time (QuantityType): dwell time per spot
            repeats (int): number of repeats
        """

        if not isinstance(dwell_time, Q_):
            raise TypeError('dwell_time must be a quantity.')

        if not has_time_dim(dwell_time):
            raise ValueError('dwell_time must have dimension [time].')

        if not isinstance(repeats, int):
            raise TypeError('repats must be an int')

        if repeats < 1:
            raise ValueError('repeats must be at least 1.')
        
        def dwell_func(point: np.ndarray) -> QuantityType:
            return dwell_time

        super().__init__(dwell_time=dwell_func, repeats=repeats)

    @property
    def dwell_time(self) -> QuantityType:
        return self['dwell_time']

    @property
    def repeats(self) -> int:
        return self['repeats']


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


class SpecialMill(MillBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


