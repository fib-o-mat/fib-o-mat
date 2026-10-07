"""Helpers of the marshmallow schemas of the NPVE step and repeat format."""
from marshmallow import Schema, fields, post_dump


class OrderedSchema(Schema):
    class Meta:
        ordered = True


class EnvelopeBaseSchema(OrderedSchema):
    __envelope__ = None

    @post_dump(pass_collection=False)
    def wrap_with_envelope(self, data, many):
        return {self.__envelope__: data}


class LCBool(fields.Field):
    """Field that serializes to a string of numbers and deserializes
    to a list of numbers.
    """

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            raise RuntimeError
        return "true" if value else "false"

    def _deserialize(self, value, attr, data, **kwargs):
        raise NotImplementedError
