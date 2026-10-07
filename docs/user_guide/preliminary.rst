Preliminary
===========

Vectors
-------

fib-o-mat provides a :class:`~fibomat.linalg.Vector` class which represents a two-dimensional mathematical vector or a point in Euclidean space. A vector can be constructed in various ways:

    - ``Vector()`` creates a null vector
    - ``Vector(x=1, y=2) = Vector(1, 2)`` creates a Cartesian vector with `(1, 2)` as components
    - ``Vector(np.array([1, 2]))`` creates the same vector as above from a numpy array
    - ``Vector([1, 2])`` creates the same vector as above from a list
    - ``Vector((1, 2))`` creates the same vector as above from a tuple
    - ``Vector(r=1, phi=np.pi)`` creates a vector from polar coordinates
    - ``Vector(Vector(1, 2))`` copies the passed vector object

Wherever a vector is expected, such a "vector-like" object (e.g. a tuple) can be passed instead.

:class:`~fibomat.linalg.Vector` supports the usual mathematical operations ::

    u = Vector(1, 2)
    v = Vector(3, 4)

    print(u + v)  # results in Vector(4, 6)
    print(u - v)  # results in Vector(-2, -2)
    print(4 * u)  # results in Vector(4, 8)
    print(u / 2)  # results in Vector(0.5, 1)

    print(u + (1, 2))  # results in Vector(2, 4)
    print(u + [1, 2])  # results in Vector(2, 4)
    print(u + np.array((1, 2)))  # results in Vector(2, 4)

    print(u.dot(v))  # prints the dot product of u and v, in this case 11

Vectors can be accessed component-wise ::

    print(u.x, u.y)  # prints "1, 2"
    print(u[0], u[1])  # prints "1, 2", too. u[0] = u.x, u[1] = u.y

    w = Vector(r=1, phi=np.pi)
    print(w.r, w.phi)  # prints "1, 3.14159"

.. warning:: Vector values cannot be changed by accessing their elements, e.g. ``u.x = 5`` does not work.
             To change a component, a new :class:`~fibomat.linalg.Vector` must be constructed:
             ``new_u = Vector(x=5, y=u.y)``.

Further properties of vectors can be accessed ::

    print(v.length)  # prints the norm (length) of the vector
    print(v.angle_about_x_axis)  # prints the angle between the vector and the positive x-axis. The result is in [0, 2pi)

    print(u.close_to(v))  # returns True if u is nearly v and otherwise False.

    print(angle_between(u, v))  # prints the angle between u and v

Some other operations which can be applied to vectors ::

    u_rot = u.rotated(np.pi / 2)  # rotates the vector counterclockwise by pi/2 about the origin
    u_mir = u.mirrored([1, 0])  # mirrors the vector at the x-axis
    u_norm = u.normalized()  # returns a vector pointing in the same direction as `u` but with the length 1

:class:`~fibomat.linalg.Vector` can be converted to a numpy array with ::

    np_array = np.asarray(u)

Vectors with units are described by the :class:`~fibomat.linalg.DimVector` class, see the next section. It supports the same functions and methods as the :class:`~fibomat.linalg.Vector` class.

All the code snippets above are combined in `examples/vectors.py <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/vectors.py>`__.

.. note:: Not all methods of vectors are introduced above. Please consult the :doc:`API reference <../api/linalg>` for all available functions and methods.


Physical units
--------------

fib-o-mat uses the `pint library <https://github.com/hgrecco/pint>`__ to represent physical units. All functionality is encapsulated in the :mod:`~fibomat.units` submodule.

Dimensioned values are created by multiplying a number with a unit, which is created by :func:`~fibomat.units.unit` ::

    from fibomat.units import unit

    length_unit = unit('µm')
    dose_unit = unit('ions / nm**2')

    length = 1 * unit('nm')  # a DimFloat
    dose = 10 * unit('ions / nm**2')

    another_length = 10 * length_unit  # equal to 10 * unit('µm')

The resulting :class:`~fibomat.units.DimFloat` objects can be added and subtracted if they have the same dimension, and multiplied and divided by numbers and other dimensioned values.

Dimensioned vectors are created from a vector and a unit, or from dimensioned values ::

    from fibomat.linalg import DimVector, Vector

    # two ways to create a dimensioned vector
    dim_vector = Vector(3, 4) * unit('µm')
    dim_vector2 = DimVector(3 * unit('µm'), 4 * unit('µm'))

.. note:: A tuple cannot be multiplied with a unit: ``(3, 4) * unit('µm')`` raises an error. Use ``Vector(3, 4) * unit('µm')``.

Values can be scaled to other units and scale factors can be calculated ::

    from fibomat.units import scale_factor, scale_to

    length_in_um = scale_to(unit('µm'), length)  # NOTE: length_in_um is a float now and NOT a dimensioned value anymore

    nm_to_um = scale_factor(unit('µm'), unit('nm'))  # scale factor (float) to scale from nm to µm

Alternatively, the methods of :class:`~fibomat.units.DimFloat` can be used ::

    length_in_um = length.m_as('µm')  # identical to scale_to(unit('µm'), length)
    length_in_nm = length.to('nm')  # a DimFloat in nm

.. note:: Older versions of fib-o-mat used ``Q_`` and ``U_`` for quantities and units. ``U_`` is an alias of :func:`~fibomat.units.unit` and ``Q_`` is deprecated.

The code of this section can be found in `examples/units.py <https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/units.py>`__.

Immutability
------------

Most classes of fib-o-mat are immutable by convention. This means that, once an object is constructed, it should not be changed anymore. All methods which "change" an object return a new object.

For example, the x component of a :class:`~fibomat.linalg.Vector` can be read but not set, and the :meth:`~fibomat.linalg.Vector.rotated` method returns a rotated :class:`~fibomat.linalg.Vector` but does not change the original one. The same is true for the transformations of shapes, patterns, sites and arrangements.

The classes which collect other objects, namely :class:`~fibomat.layout.Layout` and :class:`~fibomat.layout.Site`, as well as the lattice builders, are mutable: patterns and sites are added to them in-place.
