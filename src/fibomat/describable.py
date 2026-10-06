"""Provides the :class:`Describable` class.

Example::

    class Thing(Describable):
        # attributes which are shared (not copied) between an object and its clones
        _shared_attributes = ('_big_immutable_config',)

    thing = Thing('a thing')
    other = thing.clone()
    other = thing.with_changed_description('another thing')
"""
from __future__ import annotations

import copy
import typing as t


__all__ = ['Describable']


SelfT = t.TypeVar('SelfT', bound='Describable')


class Describable:
    """This class handles optional descriptions in the fib-o-mat library and provides :meth:`Describable.clone`."""

    _shared_attributes: t.ClassVar[t.Tuple[str, ...]] = ()
    """Names of attributes which are **not** copied by :meth:`Describable.clone` but shared between the object and its
    clones. Use this for large objects which are never modified in-place (e.g. configurations)."""

    def __init__(self, description: t.Optional[str] = None):
        """
        Args:
            description (str, optional): description
        """
        self._description = str(description) if description else None

    def clone(self: SelfT) -> SelfT:
        """Create a deep copy of the object. Attributes listed in :attr:`Describable._shared_attributes` are shared
        with the copy instead of being copied.

        Returns:
            Describable
        """
        # objects in the memo dictionary are returned by deepcopy instead of being copied
        memo: t.Dict[int, t.Any] = {}
        for name in self._shared_attributes:
            shared = getattr(self, name)
            memo[id(shared)] = shared
        return copy.deepcopy(self, memo)

    def with_changed_description(self: SelfT, new_descr: str) -> SelfT:
        """Clone the object and set the description to `new_descr`.

        Args:
            new_descr (str): new description

        Returns:
            Describable
        """
        cloned = self.clone()
        cloned._description = str(new_descr)  # pylint: disable=protected-access
        return cloned

    @property
    def description(self) -> t.Optional[str]:
        """Description str.

        Access:
            get

        Returns:
            Optional[str]
        """
        return self._description
