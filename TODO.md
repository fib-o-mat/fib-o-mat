# TODO

Open points which are postponed on purpose.

## `mill.SILMill` and `mill.DDDMill`

- The design is undecided: a dwell time which depends on the position (`DDDMill`, `SILMill`) needs a clear contract
  (in which length unit are the positions handed to the dwell time function? vectorized or per point? which return
  type?). `Mill` has a constant dwell time and does not derive from `DDDMill` anymore.
- Both classes are only moved into their own modules (`mill/ddd_mill.py`, `mill/sil_mill.py`), they are *not* reworked
  and not tested: `SILMill` still checks its arguments with `isinstance(..., Q_)` (`Q_` is a deprecated function and
  no class anymore, so `SILMill(...)` raises a `TypeError`), uses the stateful `set_unit` and has unitless parameters
  (`min_dwell_time`).

## `raster_styles.two_d.Spiral`

- Depends on `SILMill`/`DDDMill` (see above): it evaluates `mill.dwell_time(point)` and calls `mill.set_unit`.
- Re-implements the Archimedean spiral and the arc length parametrization (the helper functions
  `rasterize_sympy_curve` and `rasterize_spiral_lambertw` are dead code); `curve_tools.fill_with_spiral` returns an
  exact `ParametricCurve` which could be rasterized with `curve_tools.rasterize` instead.
- Only the minimum needed to run with the new units is planned for now.

## Raster styles and `Mill.dwell_time`

- `Mill.dwell_time` is the constant dwell time (`DimFloat`), not a function of the position anymore. The raster styles
  (`one_d.Curve`, `zero_d.SingleSpot`, `zero_d.PreRasterized`, `scansequence._apply_scan_sequence`, ...) still call
  `mill.dwell_time(point)` and must be adapted in the raster style step.
- `mill.beam` does not exist (`ContourParallel`, `optimize`); the beam is not a part of the mills yet.

## Think about the rasterize pipeline more carefully

## Raster styles: open questions

- `Spiral` (see above) ignores `scan_sequence` and `mill.repeats`, returns `None` from its `direction` property and
  has no argument checks.
- Floating point noise: `scale_factor(unit('nm'), unit('µm'))` is `999.9999999999999` (pint converts through the
  prefixes), so converted positions are not exact (`249.99999999999997` instead of `250`).

## Backends and the new units

- Exporting a layout works up to the rasterization, but the default backends are not reworked yet: e.g. the
  `SpotListBackend` fails in `_rasterize_and_add` (`dwell_points[:, :2] += self._site_pos`, `ndarray += Vector`), other
  backends use the old unit API (`Q_`, `.m`, ...).

## Backend base class

- `BackendBase.rasterized_pattern` (shape type `RasterizedPattern`) is a relict: `RasterizedPattern` is no `Shape`, so
  a `DimShape` of it cannot exist and the method is never called.
- `smart_fib.ElyBackendMeta` declares `arc` for the custom shape `EArc` and thereby replaces the base method of
  `shapes.Arc`; it should declare a method with its own name (`earc`).
- `BokehBackend.process_pattern` duplicates the dispatch of `BackendBase` (`_dispatch`).

## Default backends

- `BokehBackend(plot_reduced_lattices=True)` is only tested with a hand-made `DimLattice`; the lattice builders
  (`arrangements`) still use the old unit API (`Q_`, `.m`).
- `PatterningDurationCalculator` does not support `NPVEMill` (dose based repeats) and `npve`'s `LineByLineOutlined` anymore
  (the old code used private attributes of the npve raster style); add it when the npve backends are reworked.
- The backends `bitmap`, `svg`, `donothing` and `smart_fib` are not imported by `default_backends` and not reworked.
- `StepAndRepeatBackend` patterns with `use_bitmap=True` need the bitmap backend, which is not reworked (tested with a replacement of
  the bitmap backend only).
- FEI stream files: `round_hfw` rounds the horizontal field width to an integer µm at the end (`np.round` wraps the whole
  expression), although its comment says it rounds up to a quarter of the order of magnitude. This is kept, because it
  defines the file name and the pattern size of existing files; decide if it should round up instead.

## What does "center" mean?

- `center` is used for different things across the package, which has to be made consistent (one meaning, documented
  in `Transformable.center`), because it decides where `rotated(..., origin='center')` and `scaled(..., origin='center')`
  transform, and what `pivot` defaults to:
  - the center of the bounding box (`ArcSpline`, `Rect`, `RasterizedPoints`, ...),
  - the mean of the centers of the elements (`Group`, `DimGroup`, lattices),
  - the position of the object itself (`Circle`, `Ellipse`, `Ring`: their center point; `Site`: the position of the
    site, which is not the center of its patterns; `Spot`: its position),
  - the mean of the vertices was the meaning for `ArcSpline` before it was changed to the bounding box.
- Decide what the meaning is (e.g. always the center of the bounding box, with the "position" kept as a separate property
  like `pivot` or `position`), check every `center` property and every use of `origin='center'` (including
  `HollowArcSpline`, `Text`, `Glyph`, `Site`, `Pattern`, `Layout` and the backends, which use the center of shapes, e.g.
  the step and repeat backend as rotation center), and add tests which state the meaning for every class.

## Arrangements

- (see "What does center mean?": the center of `Group` is the mean of the centers of the elements for now.)
- Lattices: the points which lie exactly on the boundary of the shape of `from_boundary` are not included (`contains` is
  true for interior points only). Decide if points on the boundary should be included (with a tolerance).
- Lattices: there is no custom ordering of the points anymore (the removed `predicate`); `fast_axis` only selects
  row by row or column by column. Add an `order` argument (e.g. a key function of the lattice indices or coordinates) if
  serpentine or spiral orders are needed.
- `Lattice.elements_by_uv` and `points_uv` are calculated/copied on every access; cache them if lattices with very many
  elements are used in loops.
- Docs: the scripts in `use_cases/` still use the old API (`Sample`, `U_`, ...); port them and embed their plots with `fibomat-plot`.
- Docs: the pages for `ContourParallel`, `Spiral`, `fibomat.optimize` and `fibomat.from_file` are placeholders until these are reworked.
- Docs: the plots are self-contained (about 1.2 MB each), so the built docs are large; consider sharing bokeh.js between plots.
- Docs: the version switcher (`docs/_static/switcher.json`) must get an entry for every release.
