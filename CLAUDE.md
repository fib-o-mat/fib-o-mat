# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

fib-o-mat (`fibomat`) is a Python library for generating beam patterns for focused ion beam (FIB) instruments. Geometry is modeled with shape primitives, annotated with mill/raster settings, and exported to microscope-specific formats via backends. Docs: https://fib-o-mat.readthedocs.io

## Build, install, test

- The package is built with **meson-python** and contains a C++17 extension (`libfibomat`, pybind11 + CavalierContours) in `src/libfibomat`. The git submodules `src/libfibomat/cavc` and `src/libfibomat/pybind11` must be checked out (`git submodule update --init`).
- Editable install for development: `pip install --no-build-isolation -e .[exporting,io,testing]` (the `_libfibomat` extension must be compiled; pure-Python imports of `fibomat` will fail without it). `pdm.lock` and local `.venv`/`.venv_314` virtualenvs exist.
- Tests: `pytest tests/`; single test: `pytest tests/test_mill.py::test_name`. `docs/scripts/run_tests.sh` runs coverage on `fibomat.mill`. pytest config in `pyproject.toml` is commented out.
- Lint a file/dir: `docs/scripts/lint.sh <path>` (mypy, flake8, pylint, darglint, pydocstyle). Config in `.pylintrc`.
- Bokeh measure tool (TypeScript, `bokeh-measuretool/`): `./bokeh_build.sh` builds it and copies `bokeh-measuretool.min.js` into `src/fibomat/default_backends/`. The built JS is committed.
- **Adding/removing/renaming any file under `src/fibomat` requires regenerating the per-directory `meson.build` files** — run `python write_meson.py` from the repo root. Files not listed there are not installed into the wheel.
- Version is bumped with `bump2version` (`.bumpversion.cfg`; updates `pyproject.toml` and `src/fibomat/__init__.py`).
- Docs examples live in `docs/examples/` (`run_all.sh`); the docs are Sphinx (`docs/`). Note `docs/src/` is a tracked copy of `src/`; the real source is `src/`.

## Architecture

Data model (top-level `src/fibomat/`): `Sample` → `Site`s (field of view + position) → `Pattern`s (a shape + `Mill` + raster style). `Sample.export(Backend)`, `export_multi`, `export_with_description` and `Sample.plot()` walk this tree and hand it to a backend.

- **Units**: everything user-facing is dimensioned (pint, `U_`/`Q_` in `units.py`). `_dimensioned_object.DimensionedObj` and `shapes.DimShape` wrap a value plus a unit as `(obj, unit)` tuples; internally shapes/vectors are unitless and converted at the Site/Pattern boundary.
- **`linalg`**: `Vector`/`DimVector`, `BoundingBox`, and the `Transformable`/`DimTransformable` mixins (translate/rotate/scale/mirror via `_impl_*` hooks) that shapes, patterns, sites and groups all share.
- **`shapes`**: geometric primitives (Line, Arc, Polygon, Rect, Ring, Text, ParametricCurve, ArcSpline, …). `curve_tools`, `optimize`, `layout` (lattices, groups) and `from_file` (SVG/DXF) operate on them; heavy geometry (offsetting, boolean ops, SVG import, arc splines) is delegated to the C++ extension (typed in `_libfibomat.pyi`).
- **`mill` / `raster_styles`**: `Mill` (dwell time, repeats; also `DDDMill`, `SILMill`) and rasterization strategies per dimensionality (`zero_d`, `one_d`, `two_d`, `default`). Rasterizing a pattern yields a `RasterizedPattern` / scan sequence.
- **Backends** (`backend/`, `default_backends/`): `BackendBase` has one stub method per shape type; `BackendBaseMeta` introspects which shape methods a subclass implements (via the `@shape_type` decorator), so backends only implement the shapes they support and unsupported ones raise `ShapeNotSupportedError`. Backends are registered by name in the global `registry` (done in `default_backends/__init__.py`), and `Sample.export` accepts either the class or its registered name. Bundled: Bokeh (plotting), SpotList, PatterningDurationCalculator; vendor-specific ones live in subpackages (`fei`, `npve`, `smart_fib`). Bokeh/exporting deps are optional — `default_backends/__init__.py` falls back to stubs if they are missing.

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
