"""Schemas of the sites and the file of the NPVE step and repeat format.

The field names (``data_key``), their order and the default values define the XML file which NPVE reads. They must not
be changed.
"""
from marshmallow import fields

from fibomat.default_backends.npve.step_and_repeat.utils import (
    OrderedSchema,
    LCBool,
    EnvelopeBaseSchema,
)
from fibomat.default_backends.npve.step_and_repeat.common_schemas import FIBShapeSchema


class SaROptionsSchema(OrderedSchema):
    pre_image = LCBool(data_key="PreImage")
    post_image = LCBool(data_key="PostImage")
    share_shapes = LCBool(data_key="ShareShapes")
    _apply_focus_stig = LCBool(data_key="ApplyFocusStig", dump_default=False)


class SaRStageAxis(OrderedSchema):
    _type = fields.Str(data_key="@type", dump_default="float")
    value = fields.Str(data_key="#text")


class SaRStagePosition(OrderedSchema):
    x = fields.Nested(SaRStageAxis, data_key="X")
    y = fields.Nested(SaRStageAxis, data_key="Y")
    z = fields.Nested(SaRStageAxis, data_key="Z")
    m = fields.Nested(SaRStageAxis, data_key="M")
    t = fields.Nested(SaRStageAxis, data_key="T")
    r = fields.Nested(SaRStageAxis, data_key="R")


class _SaRShapesSchema(OrderedSchema):
    shapes_list: fields.List[FIBShapeSchema] = fields.List(
        fields.Nested(FIBShapeSchema), data_key="FIBShape"
    )


class _SaRSiteSchema(OrderedSchema):
    index = fields.Int(data_key="Index")
    #
    dx = fields.Float(data_key="X")
    dy = fields.Float(data_key="Y")
    _relative = LCBool(data_key="Relative", dump_default=True)
    _user_coords = LCBool(data_key="UserCoords", dump_default=False)

    _delay = fields.Float(data_key="Delay", dump_default=0.0)
    #
    fov = fields.Float(data_key="FOV")
    #
    _shape_files = fields.Str(data_key="ShapesFile", dump_default=None)

    shapes = fields.Nested(_SaRShapesSchema, data_key="Shapes")


class _SarSitesSchema(OrderedSchema):
    sites_list: fields.List[_SaRSiteSchema] = fields.List(
        fields.Nested(_SaRSiteSchema), data_key="Site"
    )


class SaRSharedShapesSchema(OrderedSchema):
    fib_shapes: fields.List[FIBShapeSchema] = fields.List(
        fields.Nested(FIBShapeSchema), data_key="FIBShape"
    )


class SaRFileSchema(EnvelopeBaseSchema):
    __envelope__ = "StepAndRepeatPatterning"

    options = fields.Nested(SaROptionsSchema, data_key="Options")

    stage_position = fields.Nested(SaRStagePosition, data_key="StagePosition")

    shared_shapes = fields.Nested(SaRSharedShapesSchema, data_key="SharedShapes")

    sites = fields.Nested(_SarSitesSchema, data_key="Sites")
