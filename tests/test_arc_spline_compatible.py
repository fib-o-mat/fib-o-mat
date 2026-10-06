import pytest

import helpers_shapes  # noqa: F401  (makes sure fibomat.shapes.shape can be imported)
from fibomat.shapes.arc_spline import ArcSpline, ArcSplineCompatible as ReexportedArcSplineCompatible
from fibomat.shapes.arc_spline_compatible import ArcSplineCompatible
from fibomat.shapes.shape import Shape


def make_spline():
    return ArcSpline([(0, 0, 0), (1, 0, 0)], False)


class Good(ArcSplineCompatible):
    def to_arc_spline(self):
        return make_spline()


class GoodShape(helpers_shapes.Dot, ArcSplineCompatible):
    def to_arc_spline(self):
        return make_spline()


class Duck:
    """Does not derive from ArcSplineCompatible."""

    def to_arc_spline(self):
        return make_spline()


class NoMethod:
    pass


class AttributeOnly:
    to_arc_spline = None


class PropertyLikeMethod:
    @property
    def to_arc_spline(self):
        return make_spline()


class TestArcSplineCompatible:
    def test_reexport(self):
        assert ReexportedArcSplineCompatible is ArcSplineCompatible

    def test_abstract_method_is_enforced(self):
        with pytest.raises(TypeError):
            ArcSplineCompatible()

        class Forgot(ArcSplineCompatible):
            pass

        with pytest.raises(TypeError):
            Forgot()

        class ForgotShape(helpers_shapes.Dot, ArcSplineCompatible):
            pass

        with pytest.raises(TypeError):
            ForgotShape()

    def test_explicit_subclasses(self):
        assert isinstance(Good(), ArcSplineCompatible)
        assert isinstance(GoodShape(), ArcSplineCompatible)
        assert isinstance(GoodShape(), Shape)
        assert isinstance(Good().to_arc_spline(), ArcSpline)

    def test_duck_typing(self):
        assert isinstance(Duck(), ArcSplineCompatible)
        assert issubclass(Duck, ArcSplineCompatible)

    def test_not_compatible(self):
        assert not isinstance(NoMethod(), ArcSplineCompatible)
        assert not isinstance(AttributeOnly(), ArcSplineCompatible)
        assert not isinstance(PropertyLikeMethod(), ArcSplineCompatible)
        assert not isinstance(helpers_shapes.Dot(), ArcSplineCompatible)
        assert not isinstance(1, ArcSplineCompatible)
        assert not isinstance(None, ArcSplineCompatible)

    def test_shapes_without_the_method_are_not_compatible(self):
        assert not issubclass(helpers_shapes.DimShape, ArcSplineCompatible)

    def test_arc_spline_is_compatible(self):
        assert isinstance(make_spline(), ArcSplineCompatible)
        assert make_spline().to_arc_spline().vertices.tolist() == make_spline().vertices.tolist()

    def test_existing_shapes_are_compatible(self):
        from fibomat.shapes.line import Line
        from fibomat.shapes.arc import Arc

        assert isinstance(Line((0, 0), (1, 1)), ArcSplineCompatible)
        assert isinstance(Arc.from_bulge((0, 0), (1, 0), 1.), ArcSplineCompatible)
