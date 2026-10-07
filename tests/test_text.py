import copy
import math
import pickle
import string

import numpy as np
import pytest
from pyhershey.glyph_factory import glyph_factory

from fibomat.composite_shapes import DimGlyph, DimText, Glyph, HollowArcSpline, Text
from fibomat.curve_tools import self_intersections
from fibomat.linalg import BoundingBox, DimVector, Vector, mirror, rotate, scale, translate
from fibomat.shapes import DimShape, Polygon, Polyline
from fibomat.units import DimensionError, DimFloat, ureg, unit

PI = math.pi
PRINTABLE = string.printable.strip().replace('\n', '').replace('\t', '').replace('\r', '').replace('\x0b', '').replace('\x0c', '')


def hershey_paths(character, font_size=1.):
    """The (unshifted) segments of a Hershey glyph."""
    glyph = glyph_factory.from_ascii(character, 'roman_simplex')
    glyph.font_size = font_size / 21
    return [np.asarray(segment, dtype=float) for segment in glyph.segments]


def distance_to_paths(points, paths):
    """Distance of each point to the nearest segment of the paths."""
    best = np.full(len(points), np.inf)
    for path in paths:
        for start, end in zip(path[:-1], path[1:]):
            direction = end - start
            length_sq = direction @ direction
            param = np.clip(((points - start) @ direction) / length_sq, 0., 1.) if length_sq else np.zeros(len(points))
            best = np.minimum(best, np.linalg.norm(points - (start + param[:, None] * direction), axis=1))
    return best


def contours_of(shape):
    """Arrays of the contours of a polygon or a hollow arc spline: (outer, holes)."""
    if isinstance(shape, HollowArcSpline):
        return shape.boundary.vertices[:, :2], [hole.vertices[:, :2] for hole in shape.holes]
    return shape.points, []


def is_simple(points):
    spline = Polygon(points).to_arc_spline()
    return self_intersections(spline) == []


def signed_area(points):
    x, y = points.T
    return 0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1))


class TestLetterJ:
    """The stroke of a glyph with a tight curve (the 'J') was a self-intersecting polygon."""

    @pytest.mark.parametrize('width', [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6])
    def test_stroke_is_a_simple_polygon(self, width):
        text = Text('J', stroke_width=width)
        (glyph,) = text.glyphs
        assert len(glyph) == 1
        (shape,) = glyph.shapes
        assert isinstance(shape, Polygon)
        outer, holes = contours_of(shape)
        assert holes == []
        assert is_simple(outer)
        assert signed_area(outer) > 0  # counterclockwise

    @pytest.mark.parametrize('width', [0.02, 0.05, 0.1, 0.2, 0.4])
    def test_stroke_surrounds_the_path(self, width):
        text = Text('J', stroke_width=width)
        outer = contours_of(text.glyphs[0].shapes[0])[0]
        paths = [path + text.glyphs[0].origin for path in hershey_paths('J')]
        polygon = Polygon(outer).to_arc_spline()

        # every point of the path lies in the stroke ...
        for path in paths:
            assert all(polygon.contains(point) for point in path[1:-1])
        # ... and no point of the outline is farther away from the path than half of the stroke width
        assert distance_to_paths(outer, paths).max() <= width / 2 + 1e-6
        # the outline touches this distance (the stroke has the given width)
        assert distance_to_paths(outer, paths).max() == pytest.approx(width / 2, rel=1e-3)

    def test_area_of_a_thin_stroke(self):
        width = 0.02
        path = hershey_paths('J')[0]
        length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        area = Text('J', stroke_width=width).glyphs[0].shapes[0].area
        assert area == pytest.approx(length * width, rel=0.02)

    def test_all_glyphs_are_simple_polygons(self):
        for width in (0.05, 0.3):
            for character in PRINTABLE:
                text = Text(character, stroke_width=width)
                paths = [path + text.glyphs[0].origin for path in hershey_paths(character)]
                for shape in text.glyphs[0]:
                    outer, holes = contours_of(shape)
                    assert is_simple(outer), (character, width)
                    assert signed_area(outer) > 0
                    for hole in holes:
                        assert is_simple(hole), (character, width)
                        assert signed_area(hole) < 0
                    assert distance_to_paths(outer, paths).max() <= width / 2 + 1e-6, (character, width)


class TestStrokes:
    def test_total_stroke_width(self):
        # the former implementation offset every side by the stroke width (the stroke was twice as wide)
        (shape,) = Text('I', stroke_width=0.1).glyphs[0].shapes
        assert shape.area == pytest.approx(1. * 0.1)
        assert shape.bounding_box.width == pytest.approx(0.1)
        assert shape.bounding_box.height == pytest.approx(1.)

    def test_stroke_scales_with_font_size(self):
        (shape,) = Text('I', font_size=3., stroke_width=0.3).glyphs[0].shapes
        assert shape.area == pytest.approx(3. * 0.3)

    def test_closed_glyphs_have_holes(self):
        # (the former implementation joined the outer and inner contour by a slit)
        (shape,) = Text('O', stroke_width=0.1).glyphs[0].shapes
        assert isinstance(shape, HollowArcSpline)
        assert len(shape.holes) == 1
        assert shape.boundary.is_closed and shape.holes[0].is_closed
        path = hershey_paths('O')[0]
        length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        assert shape.area == pytest.approx(length * 0.1, rel=0.03)
        centre = Vector(0.5238, 0.5)
        assert not shape.contains(centre)

    def test_crossing_strokes_are_united(self):
        # the strokes of the segments of a glyph do not overlap (they would be exposed twice)
        text = Text('K', stroke_width=0.1)
        shapes = text.glyphs[0].shapes
        assert len(shapes) == 1
        paths = hershey_paths('K')
        total = sum(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) for path in paths) * 0.1
        assert shapes[0].area < total

    def test_segments_with_a_common_end_point_are_joined(self):
        # the two strokes of the 'V' are chained: one polygon without a gap on the outer side of the common end point
        (shape,) = Text('V', stroke_width=0.2).glyphs[0].shapes
        paths = [path for path in hershey_paths('V')]
        assert len(paths) == 2
        bottom = paths[0][-1] if np.allclose(paths[0][-1], paths[1][0]) or np.allclose(paths[0][-1], paths[1][-1]) else paths[0][0]
        outer = Polygon(shape.points).to_arc_spline()
        # a point just below the end point is covered thanks to the bevel join
        assert outer.contains(bottom - (0., 0.03))

    def test_junctions_of_three_strokes_are_not_chained(self):
        # the 'K' has an arm which ends in the stem: the result is still one polygon
        assert len(Text('K', stroke_width=0.1).glyphs[0].shapes) == 1

    def test_mixed_glyphs(self):
        # open and closed segments: the 'i' consists of a stem and a (closed) dot
        shapes = Text('i', stroke_width=0.05).glyphs[0].shapes
        assert len(shapes) == 2
        assert all(shape.area > 0 for shape in shapes)

    def test_without_strokes_the_glyphs_are_polylines(self):
        glyph = Text('K').glyphs[0]
        assert len(glyph) == 3
        assert all(type(shape) is Polyline for shape in glyph)
        assert all(not shape.is_closed for shape in glyph)
        assert glyph.shapes[0].points == pytest.approx(hershey_paths('K')[0])

    def test_join_and_cap_styles(self):
        # (the Hershey 'V' consists of two strokes which are joined to one polyline)
        areas = {}
        for join in ('bevel', 'miter', 'round'):
            areas[join] = Text('V', stroke_width=0.2, stroke_join=join).glyphs[0].shapes[0].area
        assert areas['bevel'] < areas['round'] < areas['miter']
        caps = {cap: Text('I', stroke_width=0.2, stroke_cap=cap).glyphs[0].shapes[0].area for cap in ('butt', 'square', 'round')}
        assert caps['butt'] < caps['round'] < caps['square']
        assert caps['square'] == pytest.approx(1.2 * 0.2)


class TestShaping:
    def test_glyphs(self):
        text = Text('Hi J')
        assert [glyph.character for glyph in text] == ['H', 'i', ' ', 'J']
        assert len(text) == 4 and len(text.glyphs) == 4
        assert [bool(glyph) for glyph in text] == [True, True, False, True]
        assert [len(glyph) for glyph in text] == [3, 2, 0, 1]
        assert all(isinstance(glyph, Glyph) for glyph in text)

    def test_glyph_positions(self):
        text = Text('HiJ')
        origins = [glyph.origin for glyph in text]
        advances = [glyph.advance_width for glyph in text]
        assert origins[0] == (0., 0.)
        assert origins[1].x == pytest.approx(advances[0]) and origins[2].x == pytest.approx(sum(advances[:2]))
        assert all(origin.y == 0. for origin in origins)
        assert advances[0] == pytest.approx(glyph_factory.from_ascii('H', 'roman_simplex').advance_width / 21)

    def test_shapes_iterator(self):
        text = Text('Hi J')
        assert len(list(text.shapes())) == sum(len(glyph) for glyph in text) == 6
        assert list(text.shapes())[0] is text.glyphs[0].shapes[0]

    def test_font_size(self):
        assert Text('H').bounding_box.height == pytest.approx(1.)
        assert Text('H', font_size=2.5).bounding_box.height == pytest.approx(2.5)
        assert Text('H', font_size=2.5).font_size == 2.5
        assert Text('ab', font_size=2.).glyphs[1].origin.x == pytest.approx(2 * Text('ab').glyphs[1].origin.x)

    def test_attributes(self):
        text = Text('abc', font_size=2., stroke_width=0.1, description='foo')
        assert (text.text, text.font_size, text.stroke_width, text.description) == ('abc', 2., 0.1, 'foo')
        assert Text('abc').stroke_width is None
        assert repr(text) == "Text('abc', font_size=2.0)"

    def test_other_fonts(self):
        assert Text('Hello', mapping='roman_complex').bounding_box.height == pytest.approx(
            Text('Hello', mapping='roman_complex', font_size=1.).bounding_box.height)
        assert Text('H', mapping='roman_complex').glyphs[0].shapes != Text('H').glyphs[0].shapes

    def test_unknown_font(self):
        with pytest.raises(ValueError, match='mapping'):
            Text('abc', mapping='foo')


class TestLines:
    def test_multiple_lines(self):
        text = Text('ab\ncd\nef', font_size=2.)
        assert text.n_lines == 3
        assert [glyph.character for glyph in text.line(1)] == ['c', 'd']
        assert [glyph.character for glyph in text.line(-1)] == ['e', 'f']
        assert [glyph.origin.y for glyph in text] == pytest.approx([0, 0, -3.2, -3.2, -6.4, -6.4])
        assert len(text) == 6
        with pytest.raises(IndexError):
            text.line(3)

    def test_blank_lines(self):
        # (an empty line crashed the former implementation)
        text = Text('a\n\nb')
        assert text.n_lines == 3
        assert text.line(1) == ()
        assert text.glyphs[1].origin.y == pytest.approx(-3.2)
        assert text.baseline_anchor('left', 1) == (0., -1.6)

    def test_leading_and_trailing_line_breaks(self):
        text = Text('\nab\n')
        assert text.n_lines == 3 and len(text) == 2
        assert text.glyphs[0].origin.y == pytest.approx(-1.6)

    def test_line_spacing_follows_the_font_size(self):
        assert Text('a\nb', font_size=3.).glyphs[1].origin.y == pytest.approx(-4.8)


class TestAlignment:
    def width(self, text):
        return sum(glyph.advance_width for glyph in text.line(0))

    def test_left(self):
        text = Text('Hi J')
        assert text.glyphs[0].origin == (0., 0.)
        assert text.baseline_anchor('left') == (0., 0.)
        assert text.baseline_anchor('right').x == pytest.approx(self.width(text))
        assert text.baseline_anchor('center').x == pytest.approx(self.width(text) / 2)

    def test_center(self):
        # (the former alignment used the mean of the glyph starts)
        text = Text('Hi J', text_alignment='center')
        assert text.baseline_anchor('center') == (0., 0.)
        assert text.baseline_anchor('left').x == pytest.approx(-self.width(text) / 2)
        assert text.baseline_anchor('right').x == pytest.approx(self.width(text) / 2)
        assert text.glyphs[0].origin.x == pytest.approx(-self.width(text) / 2)

    def test_right(self):
        # (the former alignment ignored the width of the last glyph)
        text = Text('Hi J', text_alignment='right')
        assert text.baseline_anchor('right') == (0., 0.)
        assert text.baseline_anchor('left').x == pytest.approx(-self.width(text))
        last = text.glyphs[-1]
        assert last.origin.x + last.advance_width == pytest.approx(0.)

    def test_each_line_is_aligned_separately(self):
        text = Text('a\nbbbb', text_alignment='center')
        assert text.baseline_anchor('center', 0) == (0., 0.)
        assert text.baseline_anchor('center', 1) == pytest.approx((0., -1.6))
        assert text.baseline_anchor('left', 1).x < text.baseline_anchor('left', 0).x

    def test_invalid_alignment(self):
        with pytest.raises(ValueError, match='text_alignment'):
            Text('a', text_alignment='middle')

    def test_baseline_anchor_arguments(self):
        text = Text('a\nb')
        with pytest.raises(ValueError, match='pos'):
            text.baseline_anchor('top')
        with pytest.raises(IndexError):
            text.baseline_anchor('left', 5)
        assert text.baseline_anchor('left', -1) == (0., -1.6)
        assert type(text.baseline_anchor('left')) is Vector


class TestInvalidInput:
    @pytest.mark.parametrize('text', ['', ' ', '   ', '\n', ' \n  \n', '\n\n'])
    def test_text_without_glyphs(self, text):
        with pytest.raises(ValueError, match='printable'):
            Text(text)

    def test_unsupported_characters(self):
        with pytest.raises(ValueError, match='not available'):
            Text('a°b')

    @pytest.mark.parametrize('kwargs', [
        {'font_size': 0}, {'font_size': -1}, {'font_size': np.nan}, {'font_size': np.inf},
        {'stroke_width': 0}, {'stroke_width': -0.1}, {'stroke_width': np.nan},
        {'stroke_join': 'foo'}, {'stroke_cap': 'foo'},
    ])
    def test_invalid_arguments(self, kwargs):
        kwargs.setdefault('stroke_width', 0.1)
        with pytest.raises(ValueError):
            Text('abc', **kwargs)


class TestTransformations:
    def test_bounding_box_and_center(self):
        text = Text('H J')
        bbox = text.bounding_box
        assert bbox.height == pytest.approx(1.)
        assert bbox.lower_left.y == 0. and bbox.upper_right.y == pytest.approx(1.)
        assert text.center == bbox.center
        # spaces do not enlarge the box
        assert Text('Hi   ').bounding_box == Text('Hi').bounding_box
        assert Text('H J').bounding_box.height == pytest.approx(1.)

    def test_translate(self):
        text = Text('Hi J', stroke_width=0.1)
        res = text.translated((1, -2))
        assert isinstance(res, Text) and res is not text
        assert res.bounding_box == BoundingBox(text.bounding_box.lower_left + (1, -2), text.bounding_box.upper_right + (1, -2))
        assert res.glyphs[0].origin == (1., -2.)
        assert text.glyphs[0].origin == (0., 0.)

    def test_anchors_follow_the_transformations(self):
        # (the anchors stayed at the position of the untransformed text)
        text = Text('abc\nde')
        anchors = {(pos, i): text.baseline_anchor(pos, i) for pos in ('left', 'center', 'right') for i in (0, 1)}

        def check(res, func):
            for (pos, i), anchor in anchors.items():
                assert res.baseline_anchor(pos, i) == pytest.approx(func(anchor), abs=1e-9)

        check(text.translated((1, 2)), lambda p: p + (1, 2))
        check(text.rotated(0.7, (1, 1)), lambda p: p.rotated(0.7, (1, 1)))
        check(text.scaled(2., (1, 1)), lambda p: (p - (1, 1)) * 2 + (1, 1))
        check(text.mirrored((1, 2)), lambda p: p.mirrored((1, 2)))

    def test_rotate(self):
        text = Text('H')
        res = text.rotated(PI / 2)
        assert res.bounding_box.width == pytest.approx(text.bounding_box.height)
        assert res.glyphs[0].origin == pytest.approx((0., 0.))
        assert Text('ab').rotated(PI / 2).glyphs[1].origin == pytest.approx(
            (0., Text('ab').glyphs[1].origin.x))

    def test_scale(self):
        text = Text('Hi', font_size=1., stroke_width=0.1)
        res = text.scaled(2.)
        assert res.bounding_box.height == pytest.approx(2 * text.bounding_box.height)
        assert res.font_size == 2. and res.stroke_width == pytest.approx(0.2)
        assert res.glyphs[1].advance_width == pytest.approx(2 * text.glyphs[1].advance_width)
        res = text.scaled(-2.)
        assert res.font_size == 2. and res.glyphs[1].advance_width > 0

    def test_mirror(self):
        text = Text('H')
        res = text.mirrored((1, 0))
        assert res.bounding_box == BoundingBox((text.bounding_box.lower_left.x, -1.), (text.bounding_box.upper_right.x, 0.))

    def test_transformed_chain(self):
        text = Text('Hi J', stroke_width=0.1)
        res = text.transformed(translate((1, 1)) | rotate(0.3) | scale(1.5) | mirror((1, 2)))
        assert res.bounding_box.width * res.bounding_box.height > 0
        assert sum(shape.area for shape in res.shapes()) == pytest.approx(2.25 * sum(shape.area for shape in text.shapes()))

    def test_does_not_modify_the_original(self):
        text = Text('Hi', stroke_width=0.1)
        before = text.bounding_box
        text.translated((1, 1)).rotated(1.).scaled(2.).mirrored((1, 0))
        assert text.bounding_box == before and text.baseline_anchor('left') == (0., 0.)

    def test_clone(self):
        text = Text('Hi J', stroke_width=0.1, description='foo')
        clone = text.clone()
        assert clone is not text and clone.description == 'foo' and len(clone) == len(text)
        assert clone.bounding_box == text.bounding_box
        clone._impl_translate(Vector(1, 1))
        assert text.bounding_box != clone.bounding_box
        assert copy.deepcopy(text).bounding_box == text.bounding_box


class TestGlyph:
    def test_basic(self):
        shapes = [Polyline([(0, 0), (1, 1)]), Polyline([(1, 0), (0, 1)])]
        glyph = Glyph(shapes, 'x', origin=(1, 2), advance_width=1.5, description='foo')
        assert glyph.character == 'x' and glyph.origin == (1., 2.) and glyph.advance_width == 1.5
        assert glyph.description == 'foo' and len(glyph) == 2 and bool(glyph)
        assert glyph.shapes == tuple(shapes) and list(glyph) == shapes
        assert glyph.bounding_box == BoundingBox((0, 0), (1, 1)) and glyph.center == (0.5, 0.5)
        assert repr(glyph) == "Glyph('x', n_shapes=2)"

    def test_empty_glyph(self):
        glyph = Glyph([], ' ', advance_width=0.5)
        assert not glyph and len(glyph) == 0 and glyph.origin == (0., 0.)
        with pytest.raises(ValueError):
            glyph.bounding_box
        with pytest.raises(ValueError):
            glyph.center

    def test_invalid(self):
        with pytest.raises(TypeError):
            Glyph(['foo'], 'x')
        with pytest.raises(ValueError):
            Glyph([], 'x', advance_width=np.nan)
        with pytest.raises(ValueError):
            Glyph([], 'x', origin=(np.inf, 0))

    def test_transformations(self):
        glyph = Glyph([Polyline([(0, 0), (1, 1)])], 'x', origin=(1, 2), advance_width=1.)
        res = glyph.translated((1, 1))
        assert res.origin == (2., 3.) and res.shapes[0].points[0].tolist() == [1, 1]
        assert glyph.origin == (1., 2.) and glyph.shapes[0].points[0].tolist() == [0, 0]
        assert glyph.rotated(PI / 2).origin == pytest.approx((-2., 1.))
        scaled = glyph.scaled(-2.)
        assert scaled.origin == (-2., -4.) and scaled.advance_width == 2.
        assert glyph.mirrored((1, 0)).origin == pytest.approx((1., -2.))

    def test_unit(self):
        glyph = Text('H').glyphs[0]
        dim_glyph = glyph * unit('µm')
        assert isinstance(dim_glyph, DimGlyph) and (unit('µm') * glyph).unit == dim_glyph.unit
        with pytest.raises(DimensionError):
            glyph * unit('s')
        with pytest.raises(TypeError):
            glyph * 2


class TestDimText:
    def test_creation(self):
        text = Text('Hi J', description='foo')
        dim_text = text * unit('µm')
        assert isinstance(dim_text, DimText) and (unit('nm') * text).unit == ureg.Unit('nm')
        assert dim_text.unit == ureg.Unit('µm') and dim_text.text == 'Hi J' and dim_text.n_lines == 1
        assert dim_text.description == 'foo'
        assert dim_text.text_obj is text
        assert repr(dim_text).startswith("DimText('Hi J'")
        with pytest.raises(DimensionError):
            text * unit('s')
        with pytest.raises(TypeError):
            text * 2
        with pytest.raises(TypeError):
            DimText('Hi', 'µm')

    def test_glyphs_and_shapes(self):
        dim_text = Text('Hi J', stroke_width=0.1) * unit('µm')
        glyphs = dim_text.glyphs
        assert len(dim_text) == 4 and all(isinstance(glyph, DimGlyph) for glyph in dim_text)
        assert [glyph.character for glyph in glyphs] == ['H', 'i', ' ', 'J']
        shapes = list(dim_text.shapes())
        assert all(isinstance(shape, DimShape) and shape.unit == dim_text.unit for shape in shapes)
        assert len(shapes) == sum(len(glyph) for glyph in glyphs)
        assert list(dim_text.layout_elements())[0].unit == dim_text.unit
        assert glyphs[0].shapes[0].unit == dim_text.unit
        assert len(glyphs[2]) == 0 and not glyphs[2].shapes

    def test_baseline_anchors_have_units(self):
        # (the former implementation multiplied a vector with a pint unit)
        text = Text('abc\nde', text_alignment='center')
        dim_text = text * unit('µm')
        anchor = dim_text.baseline_anchor('right', 1)
        assert isinstance(anchor, DimVector)
        assert anchor == Vector(text.baseline_anchor('right', 1)) * unit('µm')
        assert dim_text.to('nm').baseline_anchor('right', 1).vector_as('µm') == pytest.approx(
            text.baseline_anchor('right', 1))

    def test_bounding_box_and_center(self):
        dim_text = Text('H') * unit('µm')
        assert dim_text.bounding_box.height.m_as('nm') == pytest.approx(1000.)
        assert dim_text.center == Vector(*Text('H').center) * unit('µm')

    def test_transformations_with_units(self):
        text = Text('Hi', stroke_width=0.1)
        dim_text = text * unit('µm')
        res = dim_text.translated(Vector(1000, 0) * unit('nm'))
        assert isinstance(res, DimText)
        assert res.text_obj.bounding_box == BoundingBox(
            text.bounding_box.lower_left + (1, 0), text.bounding_box.upper_right + (1, 0))
        assert res.baseline_anchor('left') == Vector(1, 0) * unit('µm')
        assert dim_text.baseline_anchor('left') == Vector(0, 0) * unit('µm')
        assert dim_text.rotated(PI / 2).baseline_anchor('right').x.m_as('µm') == pytest.approx(0., abs=1e-12)
        assert dim_text.scaled(2.).bounding_box.height.m_as('µm') == pytest.approx(2 * dim_text.bounding_box.height.m_as('µm'))
        assert dim_text.mirrored(Vector(1, 0) * unit('m')).bounding_box.upper_right.y.m_as('µm') == pytest.approx(0.)

    def test_to_unit(self):
        text = Text('Hi', stroke_width=0.1)
        dim_text = text * unit('µm')
        res = dim_text.to('nm')
        assert res is not dim_text and res.unit == ureg.Unit('nm')
        assert res.bounding_box == dim_text.bounding_box
        assert res.text_obj.font_size == pytest.approx(1000.)
        assert text.font_size == 1. and dim_text.unit == ureg.Unit('µm')  # original untouched
        with pytest.raises(DimensionError):
            dim_text.to('s')

    def test_clone_and_pickle(self):
        dim_text = Text('Hi J', stroke_width=0.1) * unit('µm')
        assert dim_text.clone().bounding_box == dim_text.bounding_box
        restored = pickle.loads(pickle.dumps(dim_text))
        assert restored.unit == ureg.Unit('µm') and restored.unit._REGISTRY is ureg
        assert restored.bounding_box == dim_text.bounding_box


class TestDimGlyph:
    def test_properties(self):
        text = Text('Hi')
        dim_glyph = text.glyphs[1] * unit('µm')
        assert isinstance(dim_glyph, DimGlyph) and dim_glyph.character == 'i' and len(dim_glyph) == 2
        assert dim_glyph.glyph is text.glyphs[1]
        assert isinstance(dim_glyph.advance_width, DimFloat)
        assert dim_glyph.advance_width.m_as('nm') == pytest.approx(1000 * text.glyphs[1].advance_width)
        assert dim_glyph.origin == Vector(*text.glyphs[1].origin) * unit('µm')
        assert all(isinstance(shape, DimShape) for shape in dim_glyph)
        assert list(dim_glyph.layout_elements())[0].unit == dim_glyph.unit
        assert repr(dim_glyph).startswith("DimGlyph('i', n_shapes=2")

    def test_invalid(self):
        with pytest.raises(TypeError):
            DimGlyph('x', 'µm')
        with pytest.raises(DimensionError):
            DimGlyph(Text('H').glyphs[0], 's')

    def test_transformations(self):
        dim_glyph = Text('H').glyphs[0] * unit('µm')
        res = dim_glyph.translated(Vector(1, 2) * unit('µm'))
        assert res.origin == Vector(1, 2) * unit('µm')
        assert dim_glyph.to('nm').origin == Vector(0, 0) * unit('nm')
        assert dim_glyph.to('nm').advance_width.m_as('nm') == pytest.approx(dim_glyph.advance_width.m_as('nm'))
