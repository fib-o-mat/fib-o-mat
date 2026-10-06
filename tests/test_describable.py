import copy

import pytest

from fibomat.describable import Describable


class Config:
    """Object which counts how often it is copied."""

    copies = 0

    def __init__(self, value):
        self.value = value

    def __deepcopy__(self, memo):
        Config.copies += 1
        return Config(self.value)


class Thing(Describable):
    _shared_attributes = ('_config',)

    def __init__(self, description=None):
        super().__init__(description)
        self._config = Config(1)
        self._data = [1, 2, 3]


class Plain(Describable):
    def __init__(self, description=None):
        super().__init__(description)
        self._config = Config(1)


class TestDescription:
    def test_description(self):
        assert Describable().description is None
        assert Describable('foo').description == 'foo'
        assert Describable('').description is None
        assert Describable(None).description is None

    def test_description_is_converted_to_str(self):
        assert Describable(5).description == '5'

    def test_with_changed_description(self):
        obj = Describable('foo')
        res = obj.with_changed_description('bar')
        assert res.description == 'bar'
        assert obj.description == 'foo'
        assert res is not obj
        assert Describable().with_changed_description(7).description == '7'

    def test_with_changed_description_keeps_shared_attributes(self):
        thing = Thing()
        assert thing.with_changed_description('x')._config is thing._config


class TestClone:
    def test_clone_is_deep(self):
        plain = Plain('foo')
        clone = plain.clone()
        assert clone is not plain
        assert clone.description == 'foo'
        assert clone._config is not plain._config
        assert clone._config.value == 1

    def test_clone_type(self):
        assert type(Thing().clone()) is Thing

    def test_shared_attributes(self):
        Config.copies = 0
        thing = Thing('foo')
        clone = thing.clone()
        assert clone._config is thing._config
        assert Config.copies == 0
        # everything else is copied
        assert clone._data == thing._data and clone._data is not thing._data
        clone._data.append(4)
        assert thing._data == [1, 2, 3]
        assert clone.description == 'foo'

    def test_clone_of_clone(self):
        thing = Thing()
        assert thing.clone().clone()._config is thing._config

    def test_shared_attribute_inside_containers(self):
        thing = Thing()
        container = copy.deepcopy([thing])  # plain deepcopy still copies everything
        assert container[0]._config is not thing._config

    def test_missing_shared_attribute(self):
        class Broken(Describable):
            _shared_attributes = ('_missing',)

        with pytest.raises(AttributeError):
            Broken().clone()

    def test_subclasses_can_extend_shared_attributes(self):
        class Sub(Thing):
            _shared_attributes = Thing._shared_attributes + ('_other',)

            def __init__(self):
                super().__init__()
                self._other = Config(2)

        sub = Sub()
        clone = sub.clone()
        assert clone._config is sub._config and clone._other is sub._other
