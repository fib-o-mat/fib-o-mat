from typing import List, Tuple, Optional
import base64
import io

import numpy as np

from fibomat.units import U_, Q_, scale_to
from fibomat.linalg import VectorLike
from fibomat.raster_styles import RasterStyle, two_d, one_d, zero_d, ScanSequence
from fibomat.default_backends.npve.step_and_repeat.npve_mill import NPVEMill
from fibomat.default_backends.npve.step_and_repeat.outline import LineByLineOutlined


class _Mill:
    def _set_default_values_for_raster_style(self, raster_style: RasterStyle):
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
                raise ValueError

            self.target_du = raster_style.pitch.m_as("µm")
            self.target_dv = raster_style.pitch.m_as("µm")
            self.target_dr = raster_style.pitch.m_as("µm")
            self.target_dp = raster_style.pitch.m_as("µm")
            self.angle = 0
            self.raster_style = 0
            self.operation_id = -1

        elif isinstance(raster_style, two_d.LineByLine):
            if not isinstance(raster_style.line_style, one_d.Curve):
                raise TypeError
            if not raster_style.line_style.scan_sequence == ScanSequence.CONSECUTIVE:
                raise ValueError

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
                raise NotImplementedError
        else:
            raise TypeError("Unsupported raster style.")

    def __init__(self, mill: NPVEMill, raster_style: RasterStyle):
        if not isinstance(mill, NPVEMill):
            print(mill)
            raise TypeError("Mill must be NPVEMill.")

        self._set_default_values_for_raster_style(raster_style)

        self.num_frames = 0
        self.target_dose = 0
        self.target_time = 0

        # TODO: clean this up! fehlerbehandlung und so
        try:
            self.num_frames = mill["repeats"]
            # print('REPEATS')
            self.target_mode = 3
        except KeyError:
            # print('DOSE')
            self.target_mode = 0
            if isinstance(raster_style, zero_d.SingleSpot):
                # self.target_dose = mill['dose'].m_as('pA µs') / 200
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

        # 0: left to right
        # 90: bottom to top


def encode_image(data):
    # https://stackoverflow.com/a/58917413
    std_base64chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    custom = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"

    x = base64.b64encode(data)
    return str(x)[2:-1].translate(str(x)[2:-1].maketrans(std_base64chars, custom))


class ShapeTexture:
    def __init__(self, bitmap, rect_size: Tuple[Q_, Q_], raster_style: RasterStyle):
        if isinstance(raster_style, two_d.LineByLine):
            if not isinstance(raster_style.line_style, one_d.Curve):
                raise TypeError
            if not raster_style.line_style.scan_sequence == ScanSequence.CONSECUTIVE:
                raise ValueError

            self.du = raster_style.line_style.pitch.m_as("µm")
            self.dv = raster_style.line_pitch.m_as("µm")
        else:
            raise ValueError

        rect_width, rect_height = rect_size
        img_width, img_height = bitmap.size

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
    def __init__(
        self,
        class_: str,
        id: int,
        rotation_center: VectorLike,
        nodes: List[Tuple[VectorLike, int]],
        mill: NPVEMill,
        raster_style: RasterStyle,
        # outline: Optional[LineByLineOutlined] = None,
        shape_texture: Optional = None,
        # angle: Optional[float] = 0,
    ):
        self.class_ = class_
        self.display_id = id
        self.shape_name = class_[1:]

        # for now
        self.angle = mill._scan_direction

        self.hole = False

        self.rotation_center = {"x": rotation_center[0], "y": rotation_center[1]}

        self.nodes = {
            "nodes": [
                {"x": node[0][0], "y": node[0][1], "type": node[1]} for node in nodes
            ]
        }

        self.mill = _Mill(mill, raster_style)

        if isinstance(raster_style, LineByLineOutlined):
            # TODO: make this in a clean way

            # _outlined = LCBool(data_key='Outlined', default=False)
            # _thickness = fields.Float(data_key='Thickness', default=0.5)
            # _node_style = fields.Int(data_key='NodeStyle', default=2)
            # _stroke_style = fields.Int(data_key='StrokeStyle', default=0)
            # _pen_alignment = fields.Int(data_key='PenAlignment', default=0)
            # _outline_offset = fields.Float(data_key='OutlineOffset', default=0)
            # _direction = fields.Int(data_key='Direction', default=1)
            #
            # self.outline = {
            #     "_outlined": True,
            #     "_thickness": outline,  # factor of 0.5 !?
            #     "_node_style": 0,
            #     "_stroke_style": 0,
            #     "_pen_alignment": 1,
            #     "_outline_offset": 0,
            #     # TODO: use ScanSequence!!
            #     "_direction": 1 if self.class_ == "TRing" else 0,
            # }

            self.outline = {
                "_outlined": True,
                "_thickness": scale_to(
                    U_("µm"), raster_style._outline_offset
                ),  # TODO: difference between thickness and outline offset? # factor of 0.5 !?
                "_node_style": raster_style._outline_node_style.value,
                "_stroke_style": 0,
                "_pen_alignment": raster_style._outline_alignement.value,  # TODO alignement
                "_outline_offset": 0.0,  # TODO
                # TODO: use ScanSequence!!
                "_direction": raster_style._outline_scan_style.value,
                # "_direction": 1 if self.class_ == "TRing" else 0,
            }

        if shape_texture:
            self.has_bitmap = True
            self.shape_texture = shape_texture
        else:
            self.has_bitmap = False
