# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

Units rework: commit `14f2558`.

Rust extension and maturin build: commit `d3567de`.

### Changed
- Build system changed from meson-python to maturin (`[tool.maturin]` in `pyproject.toml`, stable ABI `abi3-py38`). CI workflows build wheels with `PyO3/maturin-action` instead of cibuildwheel. The `bokeh` build requirement was dropped.
- `offset_with_islands` orients its input itself (islands clockwise, outer curve counter-clockwise) and raises `RuntimeError` for open curves.
- `ArcSpline.visit` passes segment indices starting at 0 on every call (the C++ version kept a static counter).
- `ArcSpline.impl_translate` / `impl_mirror` / `bounding_box` / `center` raise `RuntimeError` for invalid input or empty curves.
- `requires-python` lowered from `>=3.9` to `>=3.8`; added the Python 3.8 classifier.

### Added
- Rust extension `fibomat._libfibomat` in `rust/` (pyo3 + `cavalier_contours` 0.9) with unit tests (`cargo test`, `tests/test_libfibomat.py`). Same python API as the former C++ extension (`ArcSpline`, `self_intersections`, `curve_intersections`, `combine_curves`, `offset_curve`, `offset_with_islands`, `convert_arcs_to_lines`).
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

### Removed
- The C++ extension `src/libfibomat`, its git submodules (`cavc`, `pybind11`), all `meson.build` files, `write_meson.py` and the native SVG import (not exposed before; SVGs are read with `svgelements`).
- The old tests in `tests/` (to be replaced step by step).

### Fixed
- Spelling errors in `fibomat.units` docstrings.
