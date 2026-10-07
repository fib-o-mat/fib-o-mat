"""Provide the :class:`MillBase` class."""
from __future__ import annotations

import typing as t


__all__ = ['MillBase']


class MillBase:
    """Base class of all mills.

    A mill is an immutable collection of named settings which are read with ``mill['name']``. Backends use the
    settings they need and raise a :class:`KeyError` with a helpful message if a setting is missing.

    Example:
        >>> mill = MillBase(current=1, repeats=2)
        >>> mill['repeats']
        2
    """

    def __init__(self, **kwargs: t.Any):
        """
        Args:
            **kwargs: settings of the mill
        """
        self._kwargs = dict(kwargs)

    def __repr__(self) -> str:
        settings = ', '.join(f'{key}={value!r}' for key, value in self._kwargs.items())
        return f'{self.__class__.__name__}({settings})'

    def __getitem__(self, item: str) -> t.Any:
        """Return the setting `item`.

        Args:
            item (str): name of the setting

        Returns:
            Any: value of the setting

        Raises:
            KeyError: Raised if the setting does not exist or is None.
        """
        value = self._kwargs.get(item)

        if value is None:
            raise KeyError(
                f'Mill object does not have property "{item}". '
                'Maybe you need a SpecialMill with custom properties for the used backend?'
            )

        return value
