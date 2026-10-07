# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

fib-o-mat (`fibomat`) is a Python library for generating beam patterns for focused ion beam (FIB) instruments. Geometry is modeled with shape primitives, annotated with mill/raster settings, and exported to microscope-specific formats via backends. Docs: https://fib-o-mat.readthedocs.io

## Build, install, test

- The package is built with **maturin** and contains a Rust extension (`fibomat._libfibomat`, pyo3 + the `cavalier_contours` crate) in `rust/`. The crate uses the stable ABI (`abi3-py38`), so one wheel covers all Python versions >= 3.8. Maturin is configured in `[tool.maturin]` in `pyproject.toml`.
- Editable install for development: `maturin develop --release` inside the activated virtualenv (set `VIRTUAL_ENV`, e.g. to `.venv_314`), plus `pip install -e .[exporting,io,testing]`-style extras for the optional dependencies. The `_libfibomat` extension must be compiled; imports of `fibomat.curve_tools`/`shapes` fail without it. `pdm.lock` and local `.venv`/`.venv_314` virtualenvs exist.
- Rust tests: `cargo test --manifest-path rust/Cargo.toml`.
- Tests: `pytest tests/`; single test: `pytest tests/test_mill.py::test_name`. `docs/scripts/run_tests.sh` runs coverage on `fibomat.mill`. pytest config in `pyproject.toml` is commented out.
- Lint a file/dir: `docs/scripts/lint.sh <path>` (mypy, flake8, pylint, darglint, pydocstyle). Config in `.pylintrc`.
- Bokeh measure tool (TypeScript, `src/fibomat/default_backends/measuretool.ts`): after changing it run `python scripts/build_measuretool.py` (needs node), which compiles it with bokeh's compiler to `measuretool.js`. The compiled JS is committed, so node is not needed to create plots (a test checks that it belongs to the source). Plots are saved as self-contained HTML (bokehjs is embedded); `tests/test_bokeh_browser.py` checks them in headless chromium (needs `playwright` and a chromium executable, skipped otherwise).
- Python files under `src/fibomat` are picked up by maturin automatically (no per-file build lists).
- Version is bumped with `bump2version` (`.bumpversion.cfg`; updates `pyproject.toml` and `src/fibomat/__init__.py`).
- Docs examples live in `docs/examples/` (`run_all.sh`); the docs are Sphinx (`docs/`). Note `docs/src/` is a tracked copy of `src/`; the real source is `src/`.

## Architecture

Data model (`src/fibomat/layout/`): `Layout` (the pattern design for a sample) → `Site`s (field of view + position) → `Pattern`s (a shape + `Mill` + raster style). `Layout.export(BackendClass)`, `export_multi`, `export_with_description` and `Layout.plot()` walk this tree and hand it to a backend.

- **Units**: everything user-facing is dimensioned (pint, `U_`/`Q_` in `units.py`). `_dimensioned_object.DimensionedObj` and `shapes.DimShape` wrap a value plus a unit as `(obj, unit)` tuples; internally shapes/vectors are unitless and converted at the Site/Pattern boundary.
- **`linalg`**: `Vector`/`DimVector`, `BoundingBox`, and the `Transformable`/`DimTransformable` mixins (translate/rotate/scale/mirror via `_impl_*` hooks) that shapes, patterns, sites and groups all share.
- **`composite_shapes`**: shapes composed of other shapes (`HollowArcSpline`, `Ring`, `Text` with `Glyph`s); they depend on `curve_tools` and are imported after the primitives of `shapes` (which must not import `curve_tools` or `arrangements`).
- **`shapes`**: geometric primitives (Line, Arc, Polygon, Rect, Ring, Text, ParametricCurve, ArcSpline, …). `curve_tools`, `optimize`, `arrangements` (lattices, groups) and `from_file` (SVG/DXF) operate on them; heavy geometry (offsetting, boolean ops, SVG import, arc splines) is delegated to the Rust extension (`rust/src`, typed in `_libfibomat.pyi`; SVG import is done in Python via `svgelements`).
- **`mill` / `raster_styles`**: `Mill` (dwell time, repeats; also `DDDMill`, `SILMill`) and rasterization strategies per dimensionality (`zero_d`, `one_d`, `two_d`, `default`). Rasterizing a pattern yields a `RasterizedPattern` / scan sequence.
- **Backends** (`backend/`, `default_backends/`): `BackendBase` has one stub method per shape type (declared with the `@shape_type` decorator); a backend supports the shapes whose methods it overrides, unsupported ones raise `ShapeNotSupportedError` (with a fallback to the base classes of the shape, e.g. `Polygon` -> `Polyline`). Backends are passed as classes to `Layout.export` (there is no registry). Bundled: Bokeh (plotting, self-contained HTML), SpotList, PatterningDurationCalculator; vendor-specific ones live in subpackages (`fei`, `npve`, `smart_fib`; not reworked yet). The Bokeh/exporting deps are optional — `default_backends/__init__.py` falls back to a placeholder `BokehBackend` if they are missing.

Optional dependency groups: `exporting`, `io`, `gui` (beam simulation, `beam_simulation` entry point), `docs`, `testing`, `dev`.

## Conventions

- Use Google-style docstrings.
- Code must be Python 3.8 compatible.
- One class per file by default; several small, closely related classes (e.g. the `Dimension` tags in `units/dimensions.py`) may share a file.
- Do not assume anything. When in doubt, ask.

## Rework workflow

The package is being reworked step by step. The user names the files to touch in each step; stay within them.

- Record every change in `CHANGELOG.md` at the repo root (Unreleased section).
- Add type hints (3.8 compatible), useful comments and docstrings to classes and functions, and module-level code examples where appropriate.
- Check the touched files for spelling errors (in comments, docstrings and identifiers) and fix them.
- Add unit tests for each step. These replace the existing tests in `tests/`.
- If a change breaks the public API, ask the user how to proceed before making it.
- Do not touch `docs/` for now.

## Repo notes

- Untracked local clutter (`foo/`, `fibomat.tar.xz`, `.zed/`) is not part of the project.
- `tests/stubs.py` holds mock `Transformable`/`DimShape` classes shared by the tests.
