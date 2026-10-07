"""Tests of `fibomat.default_backends.bokeh_backend` (and its helper classes)."""
import math
import re
import warnings

import numpy as np
import PIL.Image
import pytest

from fibomat.arrangements import DimLattice
from fibomat.composite_shapes import HollowArcSpline, Ring
from fibomat.default_backends import BokehBackend, BokehImage, StubRasterStyle
from fibomat.default_backends._bokeh_site import NO_HATCH, ShapeType
from fibomat.layout import Layout
from fibomat.linalg import DimVector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d, two_d, zero_d
from fibomat.shapes import ArcSpline, Circle, Line, Polygon, Rect, RasterizedPoints, Spot
from fibomat.units import unit


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


def curve_style(pitch=0.25):
    return one_d.Curve(pitch * unit('µm'), ScanSequence.CONSECUTIVE)


def area_style():
    return two_d.LineByLine(0.25 * unit('µm'), ScanSequence.CONSECUTIVE, 0., False, curve_style())


def mill():
    return Mill(1. * unit('ms'), 1)


def make_layout(*patterns, center=(0, 0), **layout_kwargs):
    layout = Layout(**layout_kwargs)
    site = layout.create_site(um(*center))
    for dim_shape, style in patterns:
        site.create_pattern(dim_shape, mill(), style)
    return layout


def export(*patterns, center=(0, 0), **kwargs):
    return make_layout(*patterns, center=center).export(BokehBackend, **kwargs)


def data(backend, shape_type):
    return backend._collect_plot_data(shape_type)  # pylint: disable=protected-access


class TestConstruction:
    def test_defaults(self):
        backend = BokehBackend()
        assert backend.description is None

    def test_description_of_layout(self):
        assert make_layout(description='design').export(BokehBackend).description == 'design'

    @pytest.mark.parametrize('plot_unit', [unit('nm'), 'nm'])
    def test_unit(self, plot_unit):
        backend = export((Line((0, 0), (1, 0)) * unit('µm'), curve_style()), unit=plot_unit)
        assert data(backend, ShapeType.NON_FILLED_CURVE)['x'][0][-1] == pytest.approx(1000.)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match='length'):
            BokehBackend(unit=unit('s'))
        with pytest.raises(TypeError, match='dimensioned'):
            BokehBackend(rasterize_pitch=0.01)
        with pytest.raises(ValueError, match='length'):
            BokehBackend(rasterize_pitch=1. * unit('s'))
        with pytest.raises(ValueError, match='positive'):
            BokehBackend(rasterize_pitch=0. * unit('µm'))
        with pytest.raises(TypeError):
            BokehBackend(unknown=1)

    def test_positional_arguments_are_not_allowed(self):
        with pytest.raises(TypeError):
            BokehBackend(unit('µm'))


class TestPlotData:
    def test_line_is_a_curve(self):
        backend = export((Line((0, 0), (2, 1)) * unit('µm'), curve_style()))
        curve = data(backend, ShapeType.NON_FILLED_CURVE)
        assert curve['x'] == [[0., 2.]] and curve['y'] == [[0., 1.]]
        assert curve['site_id'][0].startswith('Site 0')
        assert 'dwell time 1.0 ms, repeats 1' in curve['mill'][0]
        assert 'Curve' in curve['raster_style'][0]
        assert not data(backend, ShapeType.FILLED_CURVE)['x']

    def test_site_center_is_added(self):
        backend = export((Line((0, 0), (2, 1)) * unit('µm'), curve_style()), center=(10, 20))
        curve = data(backend, ShapeType.NON_FILLED_CURVE)
        assert curve['x'] == [[10., 12.]] and curve['y'] == [[20., 21.]]

    def test_shape_unit_is_converted(self):
        backend = export((Line((0, 0), (2000, 0)) * unit('nm'), curve_style()))
        assert data(backend, ShapeType.NON_FILLED_CURVE)['x'] == [[0., 2.]]

    def test_area_is_filled(self):
        backend = export((Rect(2, 2) * unit('µm'), area_style()))
        filled = data(backend, ShapeType.FILLED_CURVE)
        assert len(filled['x']) == 1
        assert filled['fill_alpha'] == [0.5]
        assert filled['hatch_pattern'] == [NO_HATCH]
        outline = np.array(filled['x'][0][0][0])
        assert outline.min() == pytest.approx(-1.) and outline.max() == pytest.approx(1.)

    def test_closed_curve_with_curve_style_is_not_filled(self):
        backend = export((Rect(2, 2) * unit('µm'), curve_style()))
        assert len(data(backend, ShapeType.NON_FILLED_CURVE)['x']) == 1
        assert not data(backend, ShapeType.FILLED_CURVE)['x']

    def test_closed_curves_drawn_as_lines_are_closed(self):
        # (the line has to return to its start, otherwise the last segment is missing in the plot)
        backend = export((Rect(2, 2) * unit('µm'), curve_style()), (Circle(1.) * unit('µm'), curve_style()))
        curves = data(backend, ShapeType.NON_FILLED_CURVE)
        for x, y in zip(curves['x'], curves['y']):
            assert (x[0], y[0]) == (x[-1], y[-1])
        assert len(curves['x'][0]) == 5
        # open curves are not closed
        open_curve = export((Line((0, 0), (2, 1)) * unit('µm'), curve_style()))
        assert data(open_curve, ShapeType.NON_FILLED_CURVE)['x'] == [[0., 2.]]
        # filled polygons are closed by bokeh
        filled = export((Rect(2, 2) * unit('µm'), area_style()))
        assert len(data(filled, ShapeType.FILLED_CURVE)['x'][0][0][0]) == 4

    def test_open_arc_spline_with_area_style_is_a_curve(self):
        backend = export((ArcSpline([(0, 0, 0), (1, 0, 0), (1, 1, 0)], False) * unit('µm'), area_style()))
        assert len(data(backend, ShapeType.NON_FILLED_CURVE)['x']) == 1

    def test_spot(self):
        backend = export((Spot((1, 2)) * unit('µm'), zero_d.SingleSpot()), center=(10, 0))
        spots = data(backend, ShapeType.SPOT)
        assert spots['x'] == [11.] and spots['y'] == [2.]

    def test_circle_is_approximated_with_the_given_distance(self):
        for pitch in (0.01, 0.001):
            backend = export((Circle(5.) * unit('µm'), area_style()), rasterize_pitch=pitch * unit('µm'))
            filled = data(backend, ShapeType.FILLED_CURVE)
            outline = np.array([filled['x'][0][0][0], filled['y'][0][0][0]]).T
            assert np.allclose(np.linalg.norm(outline, axis=1), 5.)
            middles = (outline + np.roll(outline, -1, axis=0)) / 2
            assert 5. - np.linalg.norm(middles, axis=1).min() <= pitch * (1 + 1e-6)

    def test_small_arcs_do_not_fail(self):
        # an arc which is smaller than the rasterize pitch (this used to raise a RuntimeError)
        backend = export((Circle(0.0001) * unit('µm'), area_style()), rasterize_pitch=0.01 * unit('µm'))
        assert len(data(backend, ShapeType.FILLED_CURVE)['x']) == 1

    def test_hollow_arc_spline_and_ring(self):
        hollow = HollowArcSpline(
            ArcSpline([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)], True),
            [ArcSpline([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], True)]
        )
        backend = export((hollow * unit('µm'), area_style()), (Ring(3., 1.) * unit('µm'), area_style()))
        filled = data(backend, ShapeType.FILLED_CURVE)
        assert len(filled['x']) == 2
        assert all(len(entry[0]) == 2 for entry in filled['x'])  # boundary and one hole each

    def test_text_is_split_into_glyphs(self):
        from fibomat.composite_shapes import Text
        backend = export((Text('HI', font_size=1.) * unit('µm'), curve_style()))
        assert len(data(backend, ShapeType.NON_FILLED_CURVE)['x']) >= 2

    def test_rasterized_points_are_plotted_as_hatched_bounding_box(self):
        points = RasterizedPoints(np.array([[0., 0., 1.], [2., 1., 1.]]), False)
        backend = export((points * unit('µm'), zero_d.PreRasterized()))
        filled = data(backend, ShapeType.FILLED_CURVE)
        assert filled['hatch_pattern'] == ['/']
        assert max(filled['x'][0][0][0]) == pytest.approx(2.)

    def test_polygon_and_parametric_curve(self):
        from fibomat.shapes import ParametricCurve
        curve = ParametricCurve(
            lambda u: np.stack((np.cos(u), np.sin(u)), axis=-1), lambda u: np.stack((-np.sin(u), np.cos(u)), axis=-1),
            lambda u: np.stack((-np.cos(u), -np.sin(u)), axis=-1), (0., math.pi)
        )
        backend = export(
            (Polygon([(0, 0), (1, 0), (1, 1)]) * unit('µm'), area_style()), (curve * unit('µm'), curve_style())
        )
        assert len(data(backend, ShapeType.FILLED_CURVE)['x']) == 1
        assert len(data(backend, ShapeType.NON_FILLED_CURVE)['x']) == 1

    def test_pattern_color_is_used(self):
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(
            Line((0, 0), (1, 0)) * unit('µm'), mill(), curve_style(), _color='red'
        )
        assert data(layout.export(BokehBackend), ShapeType.NON_FILLED_CURVE)['color'] == ['red']

    def test_arrangements_are_split(self):
        from fibomat.arrangements import DimGroup
        layout = Layout()
        layout.create_site(um(0, 0)).create_pattern(
            DimGroup([Line((0, 0), (1, 0)) * unit('µm'), Line((0, 1), (1, 1)) * unit('µm')]), mill(), curve_style()
        )
        assert len(data(layout.export(BokehBackend), ShapeType.NON_FILLED_CURVE)['x']) == 2

    def test_patterns_outside_of_a_site_are_not_possible(self):
        backend = BokehBackend()
        from fibomat.layout import Pattern
        with pytest.raises(RuntimeError, match='site'):
            backend.process_pattern(Pattern(Spot((0, 0)) * unit('µm'), mill(), zero_d.SingleSpot()))


class TestSites:
    def test_outline_of_the_site(self):
        layout = Layout()
        layout.create_site(um(10, 0), um(4, 2), description='first')
        site = data(layout.export(BokehBackend), ShapeType.SITE)
        assert site['x'] == [[[[8., 12., 12., 8.]]]] and site['y'] == [[[[-1., -1., 1., 1.]]]]
        assert site['site_id'][0].startswith('Site 0') and 'first' in site['site_id'][0]
        assert 'fov=(4, 2)' in site['description'][0]

    def test_automatic_fov_is_used_for_the_outline(self):
        layout = make_layout((Rect(2, 2).translated((3, 0)) * unit('µm'), area_style()), center=(10, 0))
        site = data(layout.export(BokehBackend), ShapeType.SITE)
        # the pattern reaches x = 4 relative to the site: half size 4 * fov_scale (1.1)
        assert max(site['x'][0][0][0]) == pytest.approx(10 + 4.4)
        assert min(site['x'][0][0][0]) == pytest.approx(10 - 4.4)

    def test_empty_site_without_fov_has_no_outline(self):
        layout = Layout()
        layout.create_site(um(0, 0))
        backend = layout.export(BokehBackend)
        assert not data(backend, ShapeType.SITE)['x']
        backend.plot()  # does not fail

    def test_hide_sites(self):
        layout = make_layout((Line((0, 0), (1, 0)) * unit('µm'), curve_style()))
        backend = layout.export(BokehBackend, hide_sites=True)
        glyph_names = [type(renderer.glyph).__name__ for renderer in backend.fig.renderers]
        assert glyph_names.count('MultiPolygons') == 1  # (the empty source of the filled shapes)

    def test_only_sites(self):
        layout = make_layout((Line((0, 0), (1, 0)) * unit('µm'), curve_style()))
        backend = layout.export(BokehBackend, only_sites=True)
        assert not data(backend, ShapeType.NON_FILLED_CURVE)['x']
        assert len(data(backend, ShapeType.SITE)['x']) == 1

    def test_several_sites_have_different_colors_and_indices(self):
        layout = Layout()
        for i in range(3):
            layout.create_site(um(10 * i, 0), um(2, 2))
        sites = data(layout.export(BokehBackend), ShapeType.SITE)
        assert [label.split(',')[0] for label in sites['site_id']] == ['Site 0', 'Site 1', 'Site 2']
        assert len(set(sites['color'])) == 3

    def test_without_cycle_colors(self):
        layout = Layout()
        for i in range(2):
            layout.create_site(um(10 * i, 0), um(2, 2))
        assert len(set(data(layout.export(BokehBackend, cycle_colors=False), ShapeType.SITE)['color'])) == 1


class TestAnnotations:
    def test_annotations_are_black(self):
        layout = make_layout((Line((0, 0), (1, 0)) * unit('µm'), curve_style()))
        layout.add_annotation(Rect(2, 2) * unit('µm'), filled=True, description='note')
        layout.add_annotation(Line((0, 0), (1, 1)) * unit('µm'))
        backend = BokehBackend()
        # annotations are exported by `Layout.plot`; here they are added like the plot does
        from fibomat.layout import Pattern
        for annotation in layout._annotations:  # pylint: disable=protected-access
            backend.process_pattern(Pattern(
                annotation.dim_shape, None, StubRasterStyle(2 if annotation.filled else 1), _annotation=True,
                _color=annotation.color, description=annotation.description
            ))
        filled = data(backend, ShapeType.FILLED_CURVE)
        assert filled['site_id'] == ['Annotations'] and filled['color'] == ['black']
        assert 'note' in filled['description'][0]
        assert data(backend, ShapeType.NON_FILLED_CURVE)['site_id'] == ['Annotations']

    def test_layout_plot_with_annotations(self, tmp_path):
        layout = make_layout((Line((0, 0), (1, 0)) * unit('µm'), curve_style()))
        layout.add_annotation(Rect(2, 2) * unit('µm'), filled=True, color='green')
        plotter = layout.plot(show=False, filename=tmp_path / 'plot.html')
        assert (tmp_path / 'plot.html').exists()
        assert data(plotter, ShapeType.FILLED_CURVE)['color'] == ['green']

    def test_image(self, tmp_path):
        image = BokehImage(PIL.Image.new('RGBA', (4, 2), (255, 0, 0, 255)), pixel_size=0.5, center=(1, 1))
        assert (image.width, image.height) == (2., 1.)
        assert image.data.shape == (2, 4, 4)
        assert image.bounding_box.lower_left.close_to((0, 0.5))

        layout = Layout()
        layout.create_site(um(0, 0), um(5, 5))
        layout.add_annotation(image * unit('µm'))
        plotter = layout.plot(show=False)
        assert len(plotter._images) == 1  # pylint: disable=protected-access
        assert [type(renderer.glyph).__name__ for renderer in plotter.fig.renderers].count('ImageRGBA') == 1

    def test_image_from_file(self, tmp_path):
        PIL.Image.new('RGB', (3, 3), (0, 0, 255)).save(tmp_path / 'image.png')
        assert BokehImage(tmp_path / 'image.png', pixel_size=2.).width == 6.

    def test_image_must_be_an_annotation(self):
        image = BokehImage(PIL.Image.new('RGBA', (2, 2)), pixel_size=1.)
        layout = Layout()
        layout.create_site(um(0, 0), um(5, 5)).create_pattern(image * unit('µm'), mill(), curve_style())
        with pytest.raises(RuntimeError, match='annotation'):
            layout.export(BokehBackend)

    @pytest.mark.parametrize('pixel_size', [0., -1., math.inf])
    def test_invalid_pixel_size(self, pixel_size):
        with pytest.raises(ValueError):
            BokehImage(PIL.Image.new('RGBA', (2, 2)), pixel_size=pixel_size)

    def test_image_cannot_be_transformed_except_translated(self):
        image = BokehImage(PIL.Image.new('RGBA', (2, 2)), pixel_size=1.)
        assert image.translated((1, 1)).center.close_to((1, 1))
        with pytest.raises(NotImplementedError):
            image.rotated(1.)


class TestRasterizedPlot:
    def test_points_are_plotted_with_the_site_offset(self):
        layout = make_layout((Line((0, 0), (1, 0)) * unit('µm'), curve_style()), center=(10, 5))
        backend = layout.export(BokehBackend, plot_rasterized=True)
        points = np.concatenate(backend._point_cloud)  # pylint: disable=protected-access
        assert points[:, 0].tolist() == pytest.approx([10., 10.25, 10.5, 10.75, 11.])
        assert np.allclose(points[:, 1], 5.)
        assert np.allclose(points[:, 2], 1.)  # dwell time in ms
        figure = backend.fig
        assert any(type(renderer.glyph).__name__ == 'Scatter' for renderer in figure.renderers)
        assert any(type(layout_item).__name__ == 'ColorBar' for layout_item in figure.right)

    def test_failing_rasterization_falls_back_to_the_shape(self):
        layout = Layout()
        # a spot cannot be rasterized with a curve raster style
        layout.create_site(um(0, 0), um(5, 5)).create_pattern(Spot((0, 0)) * unit('µm'), mill(), curve_style())
        with pytest.warns(UserWarning, match='Rasterization failed'):
            backend = layout.export(BokehBackend, plot_rasterized=True)
        assert len(data(backend, ShapeType.SPOT)['x']) == 1


class TestReducedLattices:
    def test_only_four_elements_and_a_hatched_outline(self):
        lattice = DimLattice.from_counts(4, 3, 3. * unit('µm'), 3. * unit('µm'), Rect(1, 1) * unit('µm'))

        layout = Layout()
        layout.create_site(um(0, 0), um(50, 50)).create_pattern(lattice, mill(), area_style())
        backend = layout.export(BokehBackend, plot_reduced_lattices=True)

        filled = data(backend, ShapeType.FILLED_CURVE)
        assert filled['hatch_pattern'].count(NO_HATCH) == 4
        assert filled['hatch_pattern'].count('x') == 1

        full = layout.export(BokehBackend)
        assert len(data(full, ShapeType.FILLED_CURVE)['x']) == 12


class TestOutput:
    @staticmethod
    def backend():
        return make_layout((Line((0, 0), (1, 0)) * unit('µm'), curve_style())).export(
            BokehBackend, title='my title'
        )

    def test_figure(self):
        backend = self.backend()
        assert backend.fig is backend.fig  # created once
        assert backend.fig.title.text == 'my title'
        assert backend.fig.xaxis[0].axis_label == 'x / µm'
        assert backend.plot() is backend.fig  # plotting again rebuilds the figure

    def test_tools(self):
        names = [type(tool).__name__ for tool in self.backend().fig.tools]
        assert names.count('HoverTool') >= 1
        for expected in ('PanTool', 'WheelZoomTool', 'ResetTool', 'SaveTool', 'BoxZoomTool', 'MeasureTool'):
            assert expected in names

    def test_measure_tool_configuration(self):
        tool = next(tool for tool in self.backend().fig.tools if type(tool).__name__ == 'MeasureTool')
        assert tool.measure_unit == 'µm'
        assert tool.source.data == {'x': [], 'y': []}

    def test_legend(self):
        backend = make_layout((Line((0, 0), (1, 0)) * unit('µm'), curve_style())).export(BokehBackend)
        assert backend.fig.legend[0].visible
        hidden = make_layout((Line((0, 0), (1, 0)) * unit('µm'), curve_style())).export(BokehBackend, legend=False)
        assert not hidden.fig.legend[0].visible

    def test_legend_is_sorted_by_site(self):
        layout = Layout()
        for i in range(12):
            layout.create_site(um(5 * i, 0)).create_pattern(Line((0, 0), (1, 0)) * unit('µm'), mill(), curve_style())
        labels = [item.label['value'] for item in layout.export(BokehBackend).fig.legend[0].items]
        indices = [int(re.match(r'Site (\d+)', label).group(1)) for label in labels]
        assert indices == sorted(indices) and len(indices) == 12

    def test_empty_layout(self):
        backend = BokehBackend()
        assert backend.fig is not None
        assert '<html' in backend.html()

    def test_html_is_self_contained(self):
        html = self.backend().html()
        assert '<script src=' not in html and '<link ' not in html
        assert 'cdn.bokeh.org' not in html
        assert 'MeasureTool' in html and 'Bokeh' in html
        assert len(html) > 500_000  # bokehjs is embedded

    def test_html_with_cdn_loads_bokeh_from_the_internet(self):
        html = self.backend().html(use_cdn=True)
        assert 'cdn.bokeh.org' in html and len(html) < 500_000

    def test_save_is_self_contained_by_default(self, tmp_path):
        self.backend().save(tmp_path / 'plot.html')
        assert 'cdn.bokeh.org' not in (tmp_path / 'plot.html').read_text(encoding='utf-8')

    def test_save_with_cdn(self, tmp_path):
        self.backend().save(tmp_path / 'plot.html', use_cdn=True)
        assert 'cdn.bokeh.org' in (tmp_path / 'plot.html').read_text(encoding='utf-8')

    def test_title_of_the_document(self):
        assert '<title>my title</title>' in self.backend().html()
        assert '<title>fib-o-mat</title>' in BokehBackend().html()

    def test_show_opens_the_browser(self, monkeypatch):
        opened = []
        monkeypatch.setattr('webbrowser.open', opened.append)
        self.backend().show()
        assert len(opened) == 1 and opened[0].startswith('file://') and opened[0].endswith('.html')
        import pathlib
        from urllib.parse import unquote, urlparse
        assert 'MeasureTool' in pathlib.Path(unquote(urlparse(opened[0]).path)).read_text(encoding='utf-8')

    def test_layout_plot(self, tmp_path, monkeypatch):
        opened = []
        monkeypatch.setattr('webbrowser.open', opened.append)
        layout = make_layout((Line((0, 0), (1, 0)) * unit('µm'), curve_style()), description='design')
        plotter = layout.plot(show=True, filename=tmp_path / 'plot.html')
        assert isinstance(plotter, BokehBackend)
        assert len(opened) == 1
        assert '<title>design</title>' in (tmp_path / 'plot.html').read_text(encoding='utf-8')

    def test_no_warnings_when_plotting(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.backend().html()
