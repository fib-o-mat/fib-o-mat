"""Tests of `fibomat.layout.layout`."""
import math
import warnings

import pytest

from fibomat.arrangements import DimGroup
from fibomat.backend import BackendBase
from fibomat.layout import Layout, Pattern, Site
from fibomat.linalg import DimVector
from fibomat.mill import Mill
from fibomat.raster_styles import ScanSequence, one_d
from fibomat.shapes import Line, Rect
from fibomat.units import unit


def um(x, y):
    return DimVector(x * unit('µm'), y * unit('µm'))


def add_line(site, x0=0, x1=1):
    return site.create_pattern(
        Line((x0, 0), (x1, 0)) * unit('µm'), Mill(1. * unit('ms'), 1),
        one_d.Curve(0.25 * unit('µm'), ScanSequence.CONSECUTIVE)
    )


class RecordingBackend(BackendBase):
    """Backend which records its arguments and the processed sites."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sites = []

    def process_site(self, site):
        self.sites.append(site)


class TestSites:
    def test_empty_layout(self):
        layout = Layout(description='design')
        assert layout.description == 'design'
        assert layout.number_of_sites == 0
        assert layout.sites == []
        assert layout.bounding_box is None
        assert 'design' in repr(layout)

    def test_create_site(self):
        layout = Layout()
        site = layout.create_site(um(1, 2), description='first')
        assert isinstance(site, Site)
        assert layout.number_of_sites == 1 and layout.sites == [site]
        assert site.description == 'first'
        assert site.center.vector_as(unit('µm')).close_to((1, 2))

    def test_create_site_with_fov(self):
        site = Layout().create_site(um(0, 0), um(10, 5))
        assert site.fov.x.m_as('µm') == 10. and site.fov.y.m_as('µm') == 5.

    def test_add_site_and_iadd(self):
        layout = Layout()
        site = Site(um(0, 0))
        layout.add_site(site)
        layout += Site(um(5, 5))
        assert layout.number_of_sites == 2 and layout.sites[0] is site

    def test_add_arrangement_of_sites(self):
        layout = Layout()
        site = Site(um(0, 0))
        layout += DimGroup([site, site.translated(um(10, 0))])
        assert layout.number_of_sites == 2

    def test_add_wrong_type(self):
        layout = Layout()
        with pytest.raises(TypeError, match='sites'):
            layout.add_site('foo')
        with pytest.raises(TypeError):
            layout.add_site(DimGroup([Rect(1, 1) * unit('µm')]))
        assert layout.number_of_sites == 0

    def test_sites_is_a_copy(self):
        layout = Layout()
        layout.create_site(um(0, 0))
        layout.sites.clear()
        assert layout.number_of_sites == 1


class TestFovScale:
    def test_default(self):
        layout = Layout()
        assert layout.fov_scale == 1.1
        assert layout.create_site(um(0, 0)).fov_scale == 1.1

    def test_scale_is_passed_to_created_sites(self):
        layout = Layout(fov_scale=1.5)
        site = layout.create_site(um(0, 0))
        assert layout.fov_scale == 1.5 and site.fov_scale == 1.5
        add_line(site, 0, 4)
        assert site.fov.x.m_as('µm') == pytest.approx(1.5 * 2 * 4)

    def test_explicit_fov_ignores_the_scale(self):
        site = Layout(fov_scale=3.).create_site(um(0, 0), um(7, 7))
        add_line(site, 0, 4)
        assert site.fov.x.m_as('µm') == 7.

    def test_added_sites_keep_their_own_scale(self):
        layout = Layout(fov_scale=3.)
        site = Site(um(0, 0), fov_scale=1.2)
        layout.add_site(site)
        assert layout.sites[0].fov_scale == 1.2

    @pytest.mark.parametrize('scale', [0.5, -1., math.inf, math.nan])
    def test_invalid_scale(self, scale):
        with pytest.raises(ValueError, match='fov_scale'):
            Layout(fov_scale=scale)

    def test_scale_is_keyword_only(self):
        with pytest.raises(TypeError):
            Layout('description', 1.2)


class TestBoundingBox:
    def test_uses_absolute_coordinates(self):
        layout = Layout()
        add_line(layout.create_site(um(10, 0)), 0, 2)
        add_line(layout.create_site(um(-10, 5)), -1, 1)
        bbox = layout.bounding_box
        assert bbox.lower_left.vector_as(unit('µm')).close_to((-11, 0))
        assert bbox.upper_right.vector_as(unit('µm')).close_to((12, 5))

    def test_empty_sites_are_ignored(self):
        layout = Layout()
        layout.create_site(um(100, 100))
        add_line(layout.create_site(um(0, 0)), 0, 2)
        assert layout.bounding_box.upper_right.vector_as(unit('µm')).close_to((2, 0))

    def test_only_empty_sites(self):
        layout = Layout()
        layout.create_site(um(1, 1))
        assert layout.bounding_box is None


class TestExport:
    @staticmethod
    def layout():
        layout = Layout(description='design')
        layout.create_site(um(0, 0), description='first')
        layout.create_site(um(5, 0), description='second')
        layout.create_site(um(9, 0))
        return layout

    def test_export_with_class(self):
        layout = self.layout()
        backend = layout.export(RecordingBackend, length_unit='µm')
        assert isinstance(backend, RecordingBackend)
        assert backend.sites == layout.sites
        assert backend.kwargs == {'length_unit': 'µm', 'description': 'design'}

    def test_description_can_be_overwritten(self):
        assert self.layout().export(RecordingBackend, description='other').kwargs['description'] == 'other'

    @pytest.mark.parametrize('backend', ['spotlist', None, object, RecordingBackend(), 'bokeh'])
    def test_backend_must_be_a_backend_class(self, backend):
        with pytest.raises(TypeError, match='class'):
            self.layout().export(backend)

    def test_backend_names_are_not_supported(self):
        with pytest.raises(TypeError, match='not registered by name'):
            self.layout().export('spotlist')

    def test_other_export_methods_check_the_backend(self):
        layout = self.layout()
        with pytest.raises(TypeError):
            layout.export_multi('spotlist')
        with pytest.raises(TypeError):
            layout.export_with_description('spotlist', {'first'})

    def test_export_multi(self):
        layout = self.layout()
        backends = layout.export_multi(RecordingBackend)
        assert len(backends) == 3
        assert [backend.sites for backend in backends] == [[site] for site in layout.sites]
        assert all(backend.kwargs['description'] == 'design' for backend in backends)

    def test_export_with_description(self):
        backend = self.layout().export_with_description(RecordingBackend, {'first', 'sec.*'})
        assert [site.description for site in backend.sites] == ['first', 'second']

    def test_export_with_description_without_match(self):
        with pytest.raises(ValueError, match='matches'):
            self.layout().export_with_description(RecordingBackend, {'nothing'})

    def test_sites_without_description_do_not_match(self):
        backend = self.layout().export_with_description(RecordingBackend, {'.*'})
        assert len(backend.sites) == 2

    def test_single_site_ignores_the_pattern_with_a_warning(self):
        site = Site(um(0, 0))
        with pytest.warns(UserWarning, match='descr_pattern'):
            backend = Layout._export(RecordingBackend, site, descr_pattern={'foo'})
        assert backend.sites == [site]

    def test_single_site_without_pattern_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Layout._export(RecordingBackend, Site(um(0, 0)))


class TestAnnotationsAndPlot:
    def test_add_annotation_type_error(self):
        with pytest.raises(TypeError):
            Layout().add_annotation(Rect(1, 1))

    def test_plot_uses_annotations_and_default_title(self, monkeypatch):
        calls = {}

        class Plotter:
            def __init__(self):
                self.patterns = []

            def process_pattern(self, ptn):
                self.patterns.append(ptn)

            def plot(self):
                calls['plot'] = True

            def save(self, filename):
                calls['save'] = filename

            def show(self):
                calls['show'] = True

        plotter = Plotter()

        def fake_export(backend_class, sites, descr_pattern=None, **kwargs):
            calls['kwargs'] = kwargs
            calls['descr_pattern'] = descr_pattern
            return plotter

        monkeypatch.setattr(Layout, '_export', staticmethod(fake_export))

        layout = Layout(description='design')
        layout.add_annotation(Rect(1, 1) * unit('µm'), filled=True, color='red', description='note')
        layout.add_annotation(Line((0, 0), (1, 1)) * unit('µm'))
        layout.plot(show=False, filename='plot.html', descr_pattern={'a'})

        assert calls['kwargs'] == {'title': 'design'}
        assert calls['descr_pattern'] == {'a'}
        assert calls['plot'] and calls['save'] == 'plot.html' and 'show' not in calls
        assert [p.raster_style.dimension for p in plotter.patterns] == [2, 1]
        assert all(isinstance(p, Pattern) and p.mill is None for p in plotter.patterns)
        assert plotter.patterns[0].kwargs == {'_annotation': True, '_color': 'red'}
        assert plotter.patterns[0].description == 'note'

    def test_plot_title_can_be_overwritten(self, monkeypatch):
        seen = {}

        class Plotter:
            def plot(self):
                pass

            def show(self):
                pass

        def fake_export(backend_class, sites, descr_pattern=None, **kwargs):
            seen.update(kwargs)
            return Plotter()

        monkeypatch.setattr(Layout, '_export', staticmethod(fake_export))
        Layout(description='design').plot(show=True, title='my title')
        assert seen == {'title': 'my title'}
