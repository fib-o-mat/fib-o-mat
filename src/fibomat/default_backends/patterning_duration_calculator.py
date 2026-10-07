"""Provides the :class:`PatterningDurationCalculator`, which estimates the patterning time of a layout.

The estimate does not rasterize the patterns: the number of dwell points is calculated from the size of the shape
(length of the outline or area) and the pitches of the raster style.

Example::

    calculator = layout.export(PatterningDurationCalculator, current=1. * unit('nA'))
    calculator.print()
    calculator.total_duration  # DimFloat
"""
from __future__ import annotations

import typing as t
import warnings

import pint  # type: ignore

from fibomat import composite_shapes, shapes
from fibomat.backend import BackendBase
from fibomat.layout.pattern import Pattern
from fibomat.layout.site import Site
from fibomat.raster_styles import one_d, two_d, zero_d
from fibomat.units import DimFloat, ureg


__all__ = ['PatterningDurationCalculator']


_CURRENT_DIMENSIONALITY = ureg.get_dimensionality('[current]')


def _format_time(duration: pint.Quantity) -> str:
    """Format a duration in seconds, minutes or hours."""
    seconds = duration.m_as('s')
    if seconds < 60.:
        shown = duration.to('s')
    elif seconds < 3600.:
        shown = duration.to('min')
    else:
        shown = duration.to('hour')
    return f'{shown:~P.2f}'


class PatterningDurationCalculator(BackendBase):
    """Estimates the patterning duration of each pattern, site and of the whole layout.

    Supported are spots (:class:`~fibomat.raster_styles.zero_d.SingleSpot`), curves
    (:class:`~fibomat.raster_styles.one_d.Curve`) and areas filled with lines
    (:class:`~fibomat.raster_styles.two_d.LineByLine` with a :class:`~fibomat.raster_styles.one_d.Curve` as line
    style). The mill must be a :class:`~fibomat.mill.Mill` (or provide the settings ``dwell_time`` and ``repeats``).
    The duration is ``number of dwell points * repeats * dwell time``.
    """

    def __init__(self, current: DimFloat[t.Any], description: t.Optional[str] = None):
        """
        Args:
            current (DimFloat): beam current, e.g. ``1. * unit('nA')``. It is only shown in the summary.
            description (str, optional): description

        Raises:
            TypeError: Raised if current is not a dimensioned value.
            ValueError: Raised if current is not a positive current.
        """
        super().__init__(description)

        if not isinstance(current, DimFloat):
            raise TypeError('current must be a dimensioned value like 1. * unit("nA").')
        if current.quantity.dimensionality != _CURRENT_DIMENSIONALITY or not current.magnitude > 0.:
            raise ValueError('current must be a positive current.')
        self._current = current

        self.durations: t.List[t.Dict[str, t.Any]] = []
        """Durations of the sites: dicts with the keys ``name`` and ``patterns`` (dicts with ``name`` and
        ``duration``, a :class:`~fibomat.units.DimFloat`)."""
        self._i_site = 0
        self._i_pattern = 0

    @property
    def current(self) -> DimFloat[t.Any]:
        """Beam current.

        Access:
            get
        """
        return self._current

    @property
    def total_duration(self) -> DimFloat[t.Any]:
        """Estimated duration of the whole layout.

        Access:
            get
        """
        total = 0. * ureg.second
        for site in self.durations:
            for pattern in site['patterns']:
                total = total + pattern['duration'].quantity
        return DimFloat(total.to('s'))

    def _site_duration(self, site: t.Dict[str, t.Any]) -> pint.Quantity:
        total = 0. * ureg.second
        for pattern in site['patterns']:
            total = total + pattern['duration'].quantity
        return total

    def table(self) -> str:
        """A table with the durations of all sites.

        Returns:
            str
        """
        def percent_to_stars(fraction: float) -> str:
            n_stars = int(fraction * 20)
            return '*' * n_stars + ' ' * (20 - n_stars)

        try:
            from prettytable import PrettyTable  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as error:
            raise ImportError(
                'The summary table needs prettytable; install fibomat with the "exporting" extra.'
            ) from error

        table = PrettyTable()
        table.field_names = ['Site', '#Patterns', 'Duration', 'Cum. duration', 'Rel. duration']
        for field in table.field_names:
            table.align[field] = 'l' if field == 'Site' else 'r'

        durations = [self._site_duration(site) for site in self.durations]
        total = sum((duration.m_as('s') for duration in durations), 0.)

        cumulative = 0. * ureg.second
        for site, duration in zip(self.durations, durations):
            cumulative = cumulative + duration
            fraction = duration.m_as('s') / total if total > 0. else 0.
            table.add_row([
                site['name'], len(site['patterns']), _format_time(duration), _format_time(cumulative),
                f'>|{percent_to_stars(fraction)}|< ({fraction:.3f})'
            ])

        return (
            f'{table}\n\n'
            f'Total duration: {_format_time(self.total_duration.quantity)} @ {self._current.quantity.to("pA"):~P.2f}'
        )

    def print(self) -> None:
        """Print :meth:`table`."""
        print(self.table())

    def process_site(self, new_site: Site) -> None:
        name = f'Site {self._i_site}'
        if new_site.description:
            name += f' {new_site.description}'

        self.durations.append({'name': name, 'patterns': []})

        self._i_site += 1
        self._i_pattern = 0

        super().process_site(new_site)

    @staticmethod
    def _curve_length(shape: t.Any) -> pint.Quantity:
        """Length of the outline of a shape (in the unit of the shape's coordinates)."""
        if hasattr(shape, 'boundary_length'):
            return float(shape.boundary_length)
        return float(shape.to_arc_spline().length)

    @staticmethod
    def _area(shape: t.Any) -> float:
        """Area of a closed shape (in the unit of the shape's coordinates)."""
        if hasattr(shape, 'area'):
            return abs(float(shape.area))
        return abs(float(shape.to_arc_spline().area))

    def _estimate(self, ptn: Pattern) -> pint.Quantity:
        """Estimate the duration of a pattern.

        Raises:
            NotImplementedError: Raised for raster styles which are not supported.
            TypeError: Raised if the mill does not have a dwell time and repeats.
        """
        style = ptn.raster_style
        shape, length_unit = ptn.dim_shape.shape, ptn.dim_shape.unit

        try:
            dwell_time = ptn.mill['dwell_time']
            repeats = ptn.mill['repeats']
        except KeyError as error:
            raise TypeError('The mill must provide the settings dwell_time and repeats.') from error
        if not isinstance(dwell_time, DimFloat):
            raise TypeError('The dwell time of the mill must be a constant (a Mill).')

        if repeats < 1:
            warnings.warn(f'Got repeats < 1 for pattern {ptn!r}.', stacklevel=3)

        if isinstance(style, zero_d.SingleSpot):
            n_points = 1.
        elif isinstance(style, one_d.Curve):
            n_points = (self._curve_length(shape) * length_unit / style.pitch.quantity).m_as('')
        elif isinstance(style, two_d.LineByLine) and isinstance(style.line_style, one_d.Curve):
            area = self._area(shape) * length_unit ** 2
            n_points = (area / (style.line_pitch.quantity * style.line_style.pitch.quantity)).m_as('')
        else:
            raise NotImplementedError(f'The duration of the raster style {style!r} cannot be estimated.')

        return n_points * repeats * dwell_time.quantity

    def _process_pattern(self, ptn: Pattern) -> None:
        duration = DimFloat(self._estimate(ptn).to('s'))

        self.durations[-1]['patterns'].append({
            'name': f'{self._i_pattern} {type(ptn.dim_shape.shape).__name__}',
            'duration': duration,
        })
        self._i_pattern += 1

    def spot(self, ptn: Pattern[shapes.Spot]) -> None:
        self._process_pattern(ptn)

    def line(self, ptn: Pattern[shapes.Line]) -> None:
        self._process_pattern(ptn)

    def polyline(self, ptn: Pattern[shapes.Polyline]) -> None:
        self._process_pattern(ptn)

    def arc(self, ptn: Pattern[shapes.Arc]) -> None:
        self._process_pattern(ptn)

    def arc_spline(self, ptn: Pattern[shapes.ArcSpline]) -> None:
        self._process_pattern(ptn)

    def parametric_curve(self, ptn: Pattern[shapes.ParametricCurve]) -> None:
        self._process_pattern(ptn)

    def polygon(self, ptn: Pattern[shapes.Polygon]) -> None:
        self._process_pattern(ptn)

    def rect(self, ptn: Pattern[shapes.Rect]) -> None:
        self._process_pattern(ptn)

    def circle(self, ptn: Pattern[shapes.Circle]) -> None:
        self._process_pattern(ptn)

    def ellipse(self, ptn: Pattern[shapes.Ellipse]) -> None:
        self._process_pattern(ptn)

    def ring(self, ptn: Pattern[composite_shapes.Ring]) -> None:
        self._process_pattern(ptn)

    def hollow_arc_spline(self, ptn: Pattern[composite_shapes.HollowArcSpline]) -> None:
        self._process_pattern(ptn)
