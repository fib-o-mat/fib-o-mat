"""The shared pint unit registry of the package, including the custom ions and dose definitions.

pint quantities of different registries do not interoperate, so only :data:`ureg` is used throughout the package.

Custom definitions:

* ``ions = elementary_charge``: ``1 ions`` is exactly one elementary charge (only charges of +1 are of interest), so
  ``(1 * ureg.coulomb).to('ions')`` is identical to ``1 coulomb / elementary_charge`` (about 6.24e18 ions).
* ``[spotdose] = [current] * [time]``: charge (ions) per spot, e.g. ``ions``.
* ``[linedose] = [current] * [time] / [length]``: charge per length, e.g. ``ions / nm``.
* ``[areadose] = [current] * [time] / [length] ** 2``: charge per area, e.g. ``ions / nm**2``.

.. note:: ``ions`` are a charge, not a count of ``[current] * [time] ** 2``: ``1 ions * s`` is *not* a spot dose.
"""
from __future__ import annotations

import pint  # type: ignore


__all__ = ['ureg']


ureg: pint.UnitRegistry = pint.UnitRegistry(system='mks')
"""pint unit registry."""

# add unit `ions` to the unit registry: 1 ions == 1 elementary_charge
ureg.define('ions = elementary_charge')

# define doses as derived dimensions
ureg.define('[spotdose] = [current] * [time]')
ureg.define('[linedose] = [current] * [time] / [length]')
ureg.define('[areadose] = [current] * [time] / [length] ** 2')
