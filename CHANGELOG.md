# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

Units rework: commit `14f2558`.

### Changed
- `requires-python` lowered from `>=3.9` to `>=3.8`; added the Python 3.8 classifier.

### Added
- `fibomat.units` is now a package (`registry`, `typing_aliases`, `dimensions`, `unit_table`, `dim_base`, `dim_float`, `unit_tag`, `legacy`, `helpers`, `checking`); the public names are unchanged and re-exported from `fibomat.units`.
- `fibomat.units`: dimension-tagged scalars `DimFloat[D]` (wrapping `pint.Quantity`), dimension tags (`Dimension`, `LengthDimension`, `TimeDimension`, `DimensionlessDimension`, `SpotDoseDimension`, `LineDoseDimension`, `AreaDoseDimension`), the extension base `DimBase`, `DimensionError` and the `@check_units` decorator for runtime dimension checks.
- `fibomat.units.unit(...)`: creates dimensioned values, e.g. `4. * unit('m')`; statically typed via generated `@overload`s (`scripts/generate_unit_overloads.py`, source table `UNIT_DIMENSIONS`).
- `scale_factor`, `scale_to`, `has_length_dim` and `has_time_dim` additionally accept `DimFloat` and `unit(...)` objects.

### Changed
- `fibomat.units.U_` is now an alias of `unit(...)` and no longer the `pint.Unit` constructor. **Breaking:** `U_('µm')` returns a unit tag (`n * U_('µm')` a `DimFloat`), not a `pint.Unit`; `isinstance(x, U_)` and pint-specific usage (e.g. `.dimensionality`) in other modules must be migrated in later steps.
- Documented the `ions = elementary_charge` unit and the `[spotdose]`, `[linedose]` and `[areadose]` dimensions in `fibomat.units.registry` (`1 ions` is one elementary charge; `1 coulomb` in ions equals `1 coulomb / elementary_charge`; `1 ions * s` is not a spot dose).
- `tests/test_units.py` rewritten for the new module.

### Deprecated
- `fibomat.units.Q_` emits a `DeprecationWarning`; use `unit(...)` / `U_(...)` instead.

### Fixed
- Spelling errors in `fibomat.units` docstrings.
