from typing import Optional

import numpy as np

import pyhershey

from fibomat.shapes.polyline import Polyline
from fibomat.shapes.polygon import Polygon
from fibomat.layout import Group, DimGroup
from fibomat.linalg import Vector, translate
from fibomat.linalg.helpers import make_perp_vector, GeomLine
from fibomat.units import U_
# from fibomat.curve_tools import


class Text(Group):
    """"""

    @staticmethod
    def _add_stroke(polyline: Polyline, stroke_width: float) -> Polygon:
        def make_normals(p1: Vector, p2: Vector):
            normal = make_perp_vector(p2 - p1).normalized_to(stroke_width)
            return normal, -1 * normal

        def make_offset_segments(p1: Vector, p2: Vector):
            normals = make_normals(p1, p2)

            length = np.linalg.norm(p1 - p2)

            return (
                GeomLine(
                    Vector((p2 + normals[0]) - (p1 + normals[0])).normalized_to(length),
                    (p1 + normals[0]),
                ),
                GeomLine(
                    Vector((p2 + normals[1]) - (p1 + normals[1])).normalized_to(length),
                    (p1 + normals[1]),
                ),
            )

        def make_points(segments_, prev_segments_):
            if segments_[0].parallel_to(prev_segments_[0]):
                return [segments_[0](0)], [segments_[1](0)]
            else:
                param = segments_[0].intersect_at_param(prev_segments_[0])

                if 0 < param < 1:
                    return [segments_[0](param)], [
                        prev_segments_[1](1),
                        segments_[1](0),
                    ]
                else:
                    param = segments_[1].intersect_at_param(prev_segments_[1])

                    return [prev_segments_[0](1), segments_[0](0)], [
                        segments_[1](param)
                    ]

        polyline_points = polyline.points

        if len(polyline_points) < 2:
            raise RuntimeError

        closed = np.allclose(polyline_points[0], polyline_points[-1])

        points_first_side = []
        points_second_side = []

        first_segments = make_offset_segments(polyline_points[0], polyline_points[1])

        if not closed:
            points_first_side.append(first_segments[0](0))
            points_second_side.append(first_segments[1](0))

        prev_segments = first_segments

        for i in range(1, len(polyline_points) - 1):
            segments = make_offset_segments(polyline_points[i], polyline_points[i + 1])

            first, second = make_points(segments, prev_segments)

            points_first_side.extend(first)
            points_second_side.extend(second)

            prev_segments = segments

        if closed:
            first, second = make_points(first_segments, prev_segments)
            points_first_side.extend(first)
            points_second_side.extend(second)

            points_first_side.insert(0, first[-1])
            points_second_side.insert(0, second[-1])
        else:
            points_first_side.append(prev_segments[0](1))
            points_second_side.append(prev_segments[1](1))

        return Polygon(points_first_side + list(reversed(points_second_side)))

    def __init__(
        self,
        text: str,
        font_size: float = 1,
        stroke_width: Optional[float] = None,
        text_alignment: Optional[str] = None,
        description: Optional[str] = None,
        mapping: Optional[str] = None,
    ):
        """
        Args:
            text (str): text
            font_size (float, optional):
                font size (directly correspond to the height of upper case letters). Default to 1.
            stroke_width (float, optional):
                the stroke width of the glyph segments. If None, the glyph segments are given by 1d polylines.
                Default to None.
            text_alignment (str, optional): the text alignment. Can be "left", "center", "right". Default to "left".
        """

        mapping = mapping or "roman_simplex"

        font_size /= 21
        advance_height = 1.6 * 21 * font_size

        if not text_alignment:
            text_alignment = "left"

        shaped_text_lines = [
            pyhershey.shape_text(
                text_line,
                advance_height=0,
                mapping=mapping,
                font_size=font_size,
                text_align=text_alignment,
            )
            for text_line in text.split("\n")
        ]

        glyph_groups = []

        y_shift = 0
        anchors = []

        for shaped_text_line in shaped_text_lines:
            for shaped_glyph in shaped_text_line:
                glyph_polylines = []

                for segment in shaped_glyph["glyph"].segments:
                    # if np.allclose(segment[0], segment[-1]):
                    #     seg_shape = Polygon(segment)
                    # else:
                    #     seg_shape = Polyline(segment)

                    if stroke_width:
                        glyph_polylines.append(
                            self._add_stroke(Polyline(segment), stroke_width)
                        )
                    else:
                        glyph_polylines.append(Polyline(segment))

                if glyph_polylines:
                    glyph_groups.append(
                        Group(glyph_polylines).transformed(
                            translate(shaped_glyph["pos"]) | translate((0, y_shift))
                        )
                    )

            anchor_left = Vector(shaped_text_line[0]["pos"]) + (0, y_shift)
            anchor_right = Vector(shaped_text_line[-1]["pos"]) + (
                shaped_text_line[-1]["glyph"].advance_width,
                y_shift,
            )
            anchor_center = Vector((anchor_left.x + anchor_right.x) / 2, y_shift)

            y_shift -= advance_height

            anchors.append(
                {"left": anchor_left, "center": anchor_center, "right": anchor_right}
            )

        self._n_lines = len(shaped_text_lines)
        self._anchors = anchors

        # glyph_group = Group(glyph_groups)

        super().__init__(glyph_groups, description)

    @property
    def n_lines(self) -> int:
        """int: number of text lines"""
        return self._n_lines

    def baseline_anchor(self, pos: str, i_line: int = 0) -> Vector:
        """Return position of certain base line points.

        Args:
            pos (str): position on the base line. Can be "left", "center" or "right".
            i_line (int, optional):
                index of the base line to be used. Can be indexed like any array (0 is the first element, -1 the last
                etc.). Default to 0.

        Returns:

        """
        return self._anchors[i_line][pos]

    def __mul__(self, other):
        if isinstance(other, U_):
            # from fibomat.layout.groups.dim_group import DimGroup
            return DimText(
                self.elements, self._anchors, other, description=self.description
            )
        raise NotImplementedError


class DimText(DimGroup):
    def __init__(self, elements, anchors, unit, description):
        super().__init__([elem * unit for elem in elements], description)

        self._anchors = anchors
        self._unit = unit

    @property
    def n_lines(self) -> int:
        return len(self._anchors)

    def baseline_anchor(self, pos: str, i_line: int = 0) -> Vector:
        return self._anchors[i_line][pos] * self._unit


# def shape_text(
#
# ) -> Group:
#     """Shape a given text.
#     This create a group containing glyphs where each glyph is given as a subgroup.
#     ``text`` can be any string but may only contain printable ASCII characters and "°". New lines can be introduced
#     with "\\n".
#
#     The pivot of the returned group is set to start of the baseline of the first text row. If ``text_align`` is "center"
#     or "right", the pivot is at the center or right side of the baseline, respectively.
#
#     Args:
#         text (str): text
#         font_size (float, optional):
#             font size (directly correspond to the height of upper case letters). Default to 1.
#         stroke_width (float, optional):
#             the stroke width of the glyph segments. If None, the glyph segments are given by 1d polylines.
#             Default to None.
#         text_alignment (str, optional): the text alignment. Can be "left", "center", "right". Default to "left".
#
#     Returns:
#         Group: grouped shaped text as
#     """
#
#     font_size /= 21
#     advance_height = 1.5 * 21 * font_size
#
#     if not text_alignment:
#         text_alignment = 'left'
#
#     shaped_glyphs = pyhershey.shape_text(
#         text, advance_height=advance_height, mapping='roman_simplex', font_size=font_size, text_align=text_alignment
#     )
#
#     glyph_groups = []
#
#     for shaped_glyph in shaped_glyphs:
#         glyph_polylines = []
#
#         for segment in shaped_glyph['glyph'].segments:
#             if stroke_width:
#                 glyph_polylines.append(_add_stroke(Polyline(segment), stroke_width))
#             else:
#                 glyph_polylines.append(Polyline(segment))
#
#         if glyph_polylines:
#             glyph_groups.append(Group(glyph_polylines).translated(shaped_glyph['pos']))
#
#     glyph_group = Group(glyph_groups)
#
#     glyph_group.pivot =
#
#     return glyph_group.translated_to((0, 0))
