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
- The subclasses of `SpotListBackend` (`fei.FEIStreamFile`, `npve.NPVETxt`) still use the old unit API: they pass `Q_`
  values as `base_dwell_time` (now a `DimFloat`) and `Q_`/`.magnitude` in their `save_impl`. The other backends (`bitmap`,
  `svg`, `donothing`, `smart_fib`, `npve.step_and_repeat`) are not imported by `default_backends` and not reworked.
