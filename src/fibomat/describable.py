"""Provides the :class:`Describable` class."""

from typing import Optional, TypeVar
import copy


T = TypeVar("T", bound="DimTransformable")  # pylint: disable=invalid-name


class Describable:
    """This class handles optional descriptions in the fib-o-mat library."""

    def __init__(self, description: Optional[str] = None):
        """
        Args:
            # description (str, optional): description
        """
        #  super().__init__()

        self._description = str(description) if description else None

    def clone(self: T) -> T:
        """Create a deepcopy of the object.

        Returns:
            Describable
        """
        return copy.deepcopy(self)

    def with_changed_description(self: T, new_descr: str) -> T:
        """Clones the object and set the description to `new_descr`.

        Args:
            new_descr: new description

        Returns:
            Describable
        """
        cloned = self.clone()
        cloned._description = str(new_descr)
        return cloned

    @property
    def description(self) -> Optional[str]:
        """Description str.

        Access:
            get

        Returns:
            Optional[str]
        """
        return self._description
