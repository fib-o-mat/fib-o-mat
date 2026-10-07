"""Models of the shapes of the NPVE step and repeat files.

The classes only hold the values which are written to the file; the file format is defined by the schemas in
``common_schemas.py``. The numerical values (including the magic numbers) must not be changed, they define the files
which NPVE reads.
"""
from __future__ import annotations

import base64
import io
import typing as t

import numpy as np

from fibomat.default_backends.npve.step_and_repeat.npve_mill import NPVEMill
from fibomat.default_backends.npve.step_and_repeat.outline import LineByLineOutlined
from fibomat.linalg import VectorLike
from fibomat.raster_styles import RasterStyle, ScanSequence, one_d, two_d, zero_d
from fibomat.units import DimFloat, scale_to, unit


__all__ = ['FIBShape', 'ShapeTexture', 'encode_image']


_MICRON = unit('µm')


class _Mill:
    """Mill settings of a shape: pitches, scan direction, dose or number of repeats."""

    def _set_default_values_for_raster_style(self, raster_style: RasterStyle) -> None:
        if isinstance(raster_style, zero_d.SingleSpot):
            self.target_du = 0.0
            self.target_dv = 0.0
            self.target_dr = 0.0
            self.target_dp = 0.0

            self.raster_style = 0
            self.angle = 0
            self.operation_id = 999  # WTF
        elif isinstance(raster_style, one_d.Curve):
            if not raster_style.scan_sequence == ScanSequence.CONSECUTIVE:
                raise ValueError('Curves must have the scan sequence ScanSequence.CONSECUTIVE.')

            self.target_du = raster_style.pitch.m_as("µm")
            self.target_dv = raster_style.pitch.m_as("µm")
            self.target_dr = raster_style.pitch.m_as("µm")
            self.target_dp = raster_style.pitch.m_as("µm")
            self.angle = 0
            self.raster_style = 0
            self.operation_id = -1

        elif isinstance(raster_style, two_d.LineByLine):
            if not isinstance(raster_style.line_style, one_d.Curve):
                raise TypeError('The line style of LineByLine must be a Curve.')
            if not raster_style.line_style.scan_sequence == ScanSequence.CONSECUTIVE:
                raise ValueError('The line style must have the scan sequence ScanSequence.CONSECUTIVE.')

            self.target_du = raster_style.line_style.pitch.m_as("µm")
            self.target_dv = raster_style.line_pitch.m_as("µm")
            self.target_dr = raster_style.line_style.pitch.m_as("µm")
            self.target_dp = raster_style.line_pitch.m_as("µm")

            self.operation_id = -1

            self.angle = raster_style.alpha - np.pi / 2
            if raster_style.invert:
                self.angle += np.pi

            if raster_style.scan_sequence == ScanSequence.CONSECUTIVE:
                self.raster_style = 0
            elif raster_style.scan_sequence == ScanSequence.SERPENTINE:
                self.raster_style = 1
            elif raster_style.scan_sequence == ScanSequence.DOUBLE_SERPENTINE:
                self.raster_style = 3
            else:
                raise NotImplementedError(f'The scan sequence {raster_style.scan_sequence} is not supported by NPVE.')
        else:
            raise TypeError("Unsupported raster style.")

    def __init__(self, mill: NPVEMill, raster_style: RasterStyle):
        """
        Args:
            mill (NPVEMill): mill
            raster_style (RasterStyle): raster style (SingleSpot, Curve or LineByLine)

        Raises:
            TypeError: Raised if the mill is no NPVEMill or the raster style is not supported.
            ValueError: Raised if the scan sequence of a curve is not supported.
            NotImplementedError: Raised if the scan sequence of the lines is not supported.
        """
        if not isinstance(mill, NPVEMill):
            raise TypeError(f"Mill must be NPVEMill, got {type(mill).__name__}.")

        self._set_default_values_for_raster_style(raster_style)

        self.num_frames = 0
        self.target_dose = 0
        self.target_time = 0

        if mill.repeats is not None:
            self.num_frames = mill.repeats
            self.target_mode = 3
        else:
            self.target_mode = 0
            if isinstance(raster_style, zero_d.SingleSpot):
                self.target_dose = (
                    mill["dose"].m_as("ions") * 0.000815981757185301 / (4e4)
                )
                self.num_frames = 1
                self.custom_endpoint = True
            elif isinstance(raster_style, one_d.Curve):
                self.target_dose = (
                    mill["dose"].m_as("nC/µm") * 10
                )  # WTF why multiply by 10?!?!?
            else:
                self.target_dose = mill["dose"].m_as("nC / µm**2")

        self.dwell_time = mill["dwell_time"].m_as("µs")


def encode_image(data: bytes) -> str:
    """Encode data with the base64 variant of NPVE (the alphabet is reordered).

    Args:
        data (bytes): data

    Returns:
        str: encoded data
    """
    # https://stackoverflow.com/a/58917413
    std_base64chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    custom = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"

    encoded = base64.b64encode(data).decode('ascii')
    return encoded.translate(encoded.maketrans(std_base64chars, custom))


class ShapeTexture:
    """A bitmap which is used as texture of a rectangle."""

    def __init__(self, bitmap: t.Any, rect_size: t.Tuple[DimFloat[t.Any], DimFloat[t.Any]], raster_style: RasterStyle):
        """
        Args:
            bitmap (PIL.Image.Image): image
            rect_size (Tuple[DimFloat, DimFloat]): width and height of the rectangle (not used by NPVE)
            raster_style (RasterStyle): raster style of the rectangle (LineByLine with a Curve as line style)

        Raises:
            ValueError: Raised if the raster style is no LineByLine or has an unsupported line style.
            TypeError: Raised if the line style is no Curve.
        """
        if isinstance(raster_style, two_d.LineByLine):
            if not isinstance(raster_style.line_style, one_d.Curve):
                raise TypeError('The line style of LineByLine must be a Curve.')
            if not raster_style.line_style.scan_sequence == ScanSequence.CONSECUTIVE:
                raise ValueError('The line style must have the scan sequence ScanSequence.CONSECUTIVE.')

            self.du = raster_style.line_style.pitch.m_as("µm")
            self.dv = raster_style.line_pitch.m_as("µm")
        else:
            raise ValueError('Textures need the raster style LineByLine.')

        self.scale_x = 0.002  # float(rect_width.m_as('µm') / img_width)
        self.scale_y = 0.002  # float(rect_height.m_as('µm') / img_height)

        # wtf !?
        self.spotsize = 0.000500000023748726

        self.original_image_src = ""

        img_data = io.BytesIO()
        bitmap.save(img_data, format="png")

        self.encoded_image = {
            "filename": "bitmap.png",
            "data": encode_image(img_data.getvalue()),
        }


class FIBShape:
    """A shape of a step and repeat file."""

    def __init__(
        self,
        class_: str,
        id: int,  # pylint: disable=redefined-builtin
        rotation_center: VectorLike,
        nodes: t.List[t.Tuple[VectorLike, int]],
        mill: NPVEMill,
        raster_style: RasterStyle,
        shape_texture: t.Optional[ShapeTexture] = None,
        outline_thickness: t.Optional[float] = None,
        angle: t.Optional[float] = None,
    ):
        """
        Args:
            class_ (str): NPVE class of the shape, e.g. "TPolygon"
            id (int): display id
            rotation_center (VectorLike): rotation center (in µm)
            nodes (List[Tuple[VectorLike, int]]): nodes (in µm) and their NPVE node types
            mill (NPVEMill): mill
            raster_style (RasterStyle): raster style
            shape_texture (ShapeTexture, optional): texture
            outline_thickness (float, optional): outline thickness (in µm) of shapes which are outlined by NPVE (rings).
                If the raster style is :class:`LineByLineOutlined`, the outline is defined by the style.
            angle (float, optional): angle of the shape. Default: the scan direction of the mill.
        """
        self.mill = _Mill(mill, raster_style)  # (checks the type of the mill)

        self.class_ = class_
        self.display_id = id
        self.shape_name = class_[1:]

        self.angle = mill.scan_direction if angle is None else angle

        self.hole = False

        self.rotation_center = {"x": rotation_center[0], "y": rotation_center[1]}

        self.nodes = {
            "nodes": [
                {"x": node[0][0], "y": node[0][1], "type": node[1]} for node in nodes
            ]
        }

        if isinstance(raster_style, LineByLineOutlined):
            self.outline = {
                "_outlined": True,
                "_thickness": scale_to(_MICRON, raster_style.outline_offset),  # TODO: difference between thickness and outline offset?
                "_node_style": raster_style.outline_node_style.value,
                "_stroke_style": 0,
                "_pen_alignment": raster_style.outline_alignement.value,
                "_outline_offset": 0.0,  # TODO
                "_direction": raster_style.outline_scan_style.value,
            }
        elif outline_thickness:
            self.outline = {
                "_outlined": True,
                "_thickness": outline_thickness,  # factor of 0.5 !?
                "_node_style": 0,
                "_stroke_style": 0,
                "_pen_alignment": 1,
                "_outline_offset": 0,
                "_direction": 1 if self.class_ == "TRing" else 0,
            }

        if shape_texture:
            self.has_bitmap = True
            self.shape_texture = shape_texture
        else:
            self.has_bitmap = False
