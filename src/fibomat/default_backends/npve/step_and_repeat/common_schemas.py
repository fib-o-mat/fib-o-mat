from marshmallow import fields

from fibomat.default_backends.npve.step_and_repeat.utils import (
    OrderedSchema,
    LCBool,
)


class _RotationCenterSchema(OrderedSchema):
    x = fields.Float(data_key="X")
    y = fields.Float(data_key="Y")

    _custom_position = LCBool(data_key="CustomPosition", dump_default=False)


class _OutlineSchema(OrderedSchema):
    _outlined = LCBool(data_key="Outlined", dump_default=False)
    _thickness = fields.Float(data_key="Thickness", dump_default=0.5)
    _node_style = fields.Int(data_key="NodeStyle", dump_default=2)
    _stroke_style = fields.Int(data_key="StrokeStyle", dump_default=0)
    _pen_alignment = fields.Int(data_key="PenAlignment", dump_default=0)
    _outline_offset = fields.Float(data_key="OutlineOffset", dump_default=0)
    _direction = fields.Int(data_key="Direction", dump_default=1)


class _Node(OrderedSchema):
    x = fields.Float(data_key="X")
    y = fields.Float(data_key="Y")
    type = fields.Int(data_key="NodeType")


class _NodesSchema(OrderedSchema):
    nodes = fields.List(fields.Nested(_Node), data_key="Node")


class _MillSchema(OrderedSchema):
    # spacing on a line (two_d.LineByLine.line_style.pitch)
    target_du = fields.Float(data_key="TargetDu")
    # spacing between lines (two_d.LineByLine.line_pitch)
    target_dv = fields.Float(data_key="TargetDv")
    target_dp = fields.Float(data_key="TargetDp")
    target_dr = fields.Float(data_key="TargetDr")

    _hrefresh = fields.Float(data_key="HRefresh", dump_default=10)
    _vrefresh = fields.Float(data_key="VRefresh", dump_default=10)
    _path_refresh = fields.Float(data_key="PathRefresh", dump_default=10)
    _full_refresh = fields.Float(data_key="FullRefresh", dump_default=10)

    # number of repeats
    num_frames = fields.Int(data_key="NumFrames")

    _num_steps = fields.Int(data_key="NumSteps", dump_default=1)

    # dose
    target_dose = fields.Float(data_key="TargetDose")

    _target_depth = fields.Float(data_key="TargetDepth", dump_default=0.0)

    # total dwell time
    target_time = fields.Float(data_key="TargetTime")

    # target mode determines if number of repeats, dose or total dwell time is fixed
    # 0 = dose
    # 1 = total dwell time
    # 2 =
    # 3 = repeats
    target_mode = fields.Int(data_key="TargetMode")

    # wtf?
    custom_endpoint = LCBool(data_key="CustomEndpoint", dump_default=False)

    # dwell time per spot
    dwell_time = fields.Float(data_key="DwellTime")

    # rotation of the scan direction
    # angle = 0   : right to left
    # angle = pi/2: bottom to top
    angle = fields.Float(data_key="Angle")

    operation_id = fields.Int(data_key="OperationID")
    _custom_operation = LCBool(data_key="CustomOperation", dump_default=False)

    _beam_id = fields.Int(data_key="BeamID", dump_default=0)
    _optimize_beam_i = LCBool(data_key="OptimizeBeamI", dump_default=False)

    # fibomat eq. is ScanSequence
    raster_style = fields.Int(data_key="RasterStyle")

    _lock_spacing_dudv = LCBool(data_key="LockSpacingDuDv", dump_default=True)
    _lock_spacing_dpdr = LCBool(data_key="LockSpacingDpDr", dump_default=True)

    _defocus = fields.Float(data_key="Defocus", dump_default=0.0)

    _delta_stig_x = fields.Float(data_key="DeltaStigX", dump_default=0.0)
    _delta_stig_y = fields.Float(data_key="DeltaStigY", dump_default=0.0)


class _EncodedImage(OrderedSchema):
    filename = fields.Str(data_key="Filename")
    data = fields.Str(data_key="Data")


class _ShapeTexture(OrderedSchema):
    _stretch_to_fit = fields.Int(data_key="StretchToFit", dump_default=1)

    _mirror_x = LCBool(data_key="MirrorX", dump_default=False)
    _mirror_y = LCBool(data_key="MirrorY", dump_default=False)

    _aspect = LCBool(data_key="MaintainAspectRatio", dump_default=True)

    _angle = fields.Float(data_key="Angle", dump_default=0)

    scale_x = fields.Float(data_key="ScaleX")
    scale_y = fields.Float(data_key="ScaleY")

    _transparency = fields.Int(data_key="Transparency", dump_default=50)

    _inverted = LCBool(data_key="Inverted", dump_default=False)

    _draw_mode = fields.Int(data_key="DrawMode", dump_default=0)

    _bitmap_mode = fields.Int(data_key="BitmapMode", dump_default=0)

    _angle_rel_to = fields.Float(data_key="AngleRelativeTo", dump_default=0)

    du = fields.Float(data_key="Du")
    dv = fields.Float(data_key="Dv")

    spotsize = fields.Float(data_key="Spotsize")

    original_image_src = fields.Str(data_key="OriginalImageSRC")

    encoded_image = fields.Nested(_EncodedImage, data_key="EncodedImage")


class FIBShapeSchema(OrderedSchema):
    class_ = fields.Str(data_key="Class")

    display_id = fields.Int(data_key="DisplayID")
    _system_id = fields.Int(data_key="SystemID", dump_default=0)
    _show_id = LCBool(data_key="ShowID", dump_default=True)
    shape_name = fields.Str(data_key="ShapeName")

    angle = fields.Float(data_key="Angle")

    _edge_index = fields.Int(data_key="EdgeIndex", dump_default=-1)

    hole = LCBool(data_key="Hole")

    _active = LCBool(data_key="Active", dump_default=True)
    _locked = LCBool(data_key="Locked", dump_default=False)

    rotation_center = fields.Nested(_RotationCenterSchema, data_key="RotationCenter")

    outline = fields.Nested(_OutlineSchema, data_key="Outline", dump_default=tuple())

    # shape_texture = fields.Nested(_ShapeTexture, data_key='ShapeTexture', dump_default=tuple())

    nodes = fields.Nested(_NodesSchema, data_key="Nodes")

    _offset = fields.Float(data_key="Offset", dump_default=0.0)

    has_bitmap = LCBool(data_key="HasBitmap")

    mill = fields.Nested(_MillSchema, data_key="Mill")
