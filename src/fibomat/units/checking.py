"""Runtime validation of dimensions (:func:`check_units`)."""
from __future__ import annotations

import functools
import inspect
import typing as t

from fibomat.units.dim_base import DimBase
from fibomat.units.dimensions import Dimension, DimensionError


__all__ = ['check_units']


def _dimension_checks(hints: t.Dict[str, t.Any]) -> t.Iterator[t.Tuple[str, type, t.Type[Dimension]]]:
    """Yield ``(name, origin_class, dimension)`` for every annotation shaped like ``SomeDimType[SomeDimension]``."""
    for name, hint in hints.items():
        origin = t.get_origin(hint)
        args = t.get_args(hint)
        if origin is None or not args:
            continue
        if not (isinstance(origin, type) and issubclass(origin, DimBase)):
            continue
        dimension = args[0]
        if isinstance(dimension, type) and issubclass(dimension, Dimension):
            yield name, origin, dimension


def _check_value(label: str, value: t.Any, origin: type, dimension: t.Type[Dimension]) -> None:
    if not isinstance(value, origin):
        raise DimensionError(f'{label!r}: expected a {origin.__name__}[{dimension.__name__}], got {type(value).__name__}')

    for suffix, quantity in value._dim_quantities():  # pylint: disable=protected-access
        if not dimension.matches(quantity):
            where = label if not suffix else f'{label}.{suffix}'
            raise DimensionError(
                f'{where!r}: expected dimension {dimension.pint_dimension!r} ({dimension.__name__}), '
                f'got {quantity.dimensionality} (units: {quantity.units})'
            )


F = t.TypeVar('F', bound=t.Callable[..., t.Any])


def check_units(func: F) -> F:
    """Decorator which validates at call time the dimensions of all arguments and the return value.

    Every parameter (and the return value) annotated as ``SomeDimType[SomeDimension]``, where ``SomeDimType`` is
    :class:`DimFloat` or any other :class:`DimBase` subclass, must hold a value whose embedded pint quantities have
    that dimensionality. This is independent of static type checking and catches values a type checker cannot verify
    (data loaded at runtime, `Any`, ``# type: ignore``, ...).

    Args:
        func (Callable): function to be wrapped

    Returns:
        Callable: wrapped function

    Raises:
        DimensionError: (raised by the wrapper) a value does not have the annotated dimension.
    """
    sig = inspect.signature(func)
    hints = t.get_type_hints(func)
    param_checks = list(_dimension_checks({k: v for k, v in hints.items() if k != 'return'}))
    return_check = next(iter(_dimension_checks({'return': hints['return']})), None) if 'return' in hints else None

    @functools.wraps(func)
    def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, origin, dimension in param_checks:
            if name in bound.arguments:
                _check_value(name, bound.arguments[name], origin, dimension)

        result = func(*args, **kwargs)

        if return_check is not None:
            _, origin, dimension = return_check
            _check_value('return value', result, origin, dimension)

        return result

    return t.cast(F, wrapper)
