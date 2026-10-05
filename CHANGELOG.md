# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

Units rework: commit `14f2558`.

Rust extension and maturin build: commit `d3567de`.

### Changed
- `fibomat.linalg.transformables`: `TransformableBase` is removed (also from `fibomat.linalg`). `Transformable` is now generic (`Transformable[VectorT, BBoxT]`), `DimTransformable` derives from `Transformable[DimVector, DimBoundingBox]`. Arguments of `translated`, `translated_to`, `mirrored`, `transformed` and the `origin` arguments are converted to the vector type of the object before the `_impl_*` methods are called.
- `fibomat.linalg.boundingboxes`: `BoundingBoxBase` is removed. `BoundingBox` is generic over the vector type and `DimBoundingBox` derives from `BoundingBox[DimVector]`. Boxes with and without units cannot be mixed (`TypeError`); `BoundingBox.corners` returns the corners in counterclockwise order (lower left, lower right, upper right, upper left); `from_points` accepts any iterable of vector-likes.
- `rotate(...)`, `scale(...)` validate their arguments (`TypeError` for non-numbers, `ValueError` for non-finite values or a zero scale factor). The `|` operator creates a new transformation chain and no longer modifies its left operand.
- `GeomLine.intersect_at_param` / `intersect_at` raise `ValueError` for parallel lines; `GeomLine.parallel_to` uses the sine of the angle (default tolerance 1e-8) instead of `cos ~ 1` (which treated angles up to ~0.26 degree as parallel).
- `fibomat.linalg.vectors`: `VectorBase` is removed. `Vector` is now generic (`Vector[S]`) and implements all vector logic; `DimVector` derives from `Vector[DimFloat[LengthDimension]]` (components, `mag`, ... are `DimFloat`, `dot`/`cross` return area `DimFloat`). `angle_between` and `signed_angle_between` moved to `vector_functions.py` (still importable from `fibomat.linalg(.vectors)`).
- `DimVector` accepts `DimFloat` components. `pint.Quantity` components are still accepted but emit a `DeprecationWarning`. `DimVector` and `Vector` can no longer be mixed in arithmetic (`VectorValueError`).
- Floating point types such as `np.float32` are accepted as vector components.
- `Vector.__eq__` returns `False` (instead of raising) for non-vector-like operands and `NotImplemented` for non-iterables; vectors are explicitly unhashable.
- `DimVector.close_to` uses an absolute tolerance of 1 pm (independent of the internal unit); `Vector.close_to` of 1e-8.
- Build system changed from meson-python to maturin (`[tool.maturin]` in `pyproject.toml`, stable ABI `abi3-py38`). CI workflows build wheels with `PyO3/maturin-action` instead of cibuildwheel. The `bokeh` build requirement was dropped.
- `offset_with_islands` orients its input itself (islands clockwise, outer curve counter-clockwise) and raises `RuntimeError` for open curves.
- `ArcSpline.visit` passes segment indices starting at 0 on every call (the C++ version kept a static counter).
- `ArcSpline.impl_translate` / `impl_mirror` / `bounding_box` / `center` raise `RuntimeError` for invalid input or empty curves.
- `requires-python` lowered from `>=3.9` to `>=3.8`; added the Python 3.8 classifier.

### Added
- Tests for `Transformable`/`DimTransformable`, the transformation builders, `BoundingBox`/`DimBoundingBox` and `fibomat.linalg.helpers` (`tests/test_transformables.py`, `tests/test_boundingboxes.py`, `tests/test_linalg_helpers.py`).
- `BoundingBox * unit('µm')` and `unit('µm') * BoundingBox` create a `DimBoundingBox`.
- `GeomLine.direction`, `GeomLine.support`, `GeomLine.__repr__`; `GeomLine.parallel_to(other, tol)`; `GeomLine.__call__` accepts arrays of parameters.
- `DimVector.from_vector`, `DimVector.to`; `Vector * unit('µm')` creates a `DimVector`.
- Tests for `Vector`, `DimVector` and the angle functions (`tests/test_vector.py`, `tests/test_dim_vector.py`).
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
- `Transformable.rotated` and `scaled` returned the object itself (not a copy) for rotation angles close to 0 and scale factors close to 1, and silently skipped small but valid transformations (`np.isclose`). They always return a transformed clone now.
- `Transformable.mirrored` did not convert its argument and accepted the null vector; the `pivot` documentation described a callable without parameters although the object is passed.
- `BoundingBox.scaled(val)` scaled correctly only for `val == 2` (each side moved by `val / 4`); width and height are now scaled by `val` about the center. Non-positive factors raise `ValueError`.
- `BoundingBox.extended` consumed generators when probing for a single vector and returned a wrong result; `BoundingBox.__eq__` raised `NotImplementedError` for non-boxes (now `False`).
- `BoundingBox` rejects non-finite coordinates; `contains` and `overlaps_with` raise `TypeError` for boxes of the other kind instead of comparing quantities and floats.
- `GeomLine`: `parallel_to` did not detect antiparallel lines; `find_param` printed debugging output, treated direction components below 1e-8 as zero and had no consistent check for tiny directions; `GeomLine.__init__` rejected short (but valid) directions (`np.allclose`); `__call__` failed for arrays of parameters.
- `fibomat.units`: `unit(...)` tags and `DimFloat` refused nothing: `unit('m') * vector` or `dim_float * vector` silently wrapped the operand into a quantity. Operands which are no numbers now yield `NotImplemented` (-> `TypeError`), comparisons and `+`/`-` with non-`DimFloat` operands raise `TypeError` instead of `AttributeError`, and numpy scalars/arrays on the left side use the reflected operators (`np.float64(2) * unit('m')` works, `np.array(...) * unit('m')` raises). `unit('µm') * Vector(1, 2)` now creates a `DimVector`, like `Vector(1, 2) * unit('µm')`.
- `DimVector` no longer modifies the quantities passed to its constructor (`y.ito(x.u)`).
- `DimVector` with nanometer-scale values in meters was treated as the null vector (`np.allclose` default tolerance); `angle_about_x_axis` checks for an exact null vector.
- `angle_between`: wrong length check, `nan` for rounding errors (cosine out of [-1, 1]), wrong results for dim-vectors with different units, silent `nan` for null vectors (now `ValueError`). `signed_angle_between` converts units, too.
- `normalized`, `normalized_to`, `projected` and `mirrored` raise `ValueError` for null vectors/axes instead of returning `nan`; division by zero raises `ZeroDivisionError`.
- `sum([...])` of vectors and `np.float64 * vector` work; `np.integer` indices are supported; `normalized_to` and `projected` work for `DimVector`.
- `DimVector` can be pickled.
- Spelling errors in `fibomat.units` docstrings.
