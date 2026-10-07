Custom rasterization style
==========================
This example shows how a custom rasterization style can be created.

In this example, a style for 1-dim is implemented where the dwell times of the rasterized spots of the shape are not equal but linearly ramped between ``ramp_start`` and ``ramp_end``.
E.g. if ``ramp_start = 1`` and ``ramp_end = 5`` and the number of rasterized spots are five, the dwell times would be
``[1, 2, 3, 4, 5] * mill.dwell_time``.

To get started, all necessary fib-o-mat modules are imported and a subclass of :class:`~fibomat.raster_styles.RasterStyle` is created with a corresponding ``__init__`` method ::

    import numpy as np

    from fibomat.raster_styles import RasterStyle
    from fibomat.units import DimFloat, has_length_dim, LengthUnit, TimeUnit, scale_to, scale_factor
    from fibomat.shapes import DimShape
    from fibomat.mill import Mill
    from fibomat.rasterizedpattern import RasterizedPattern
    from fibomat.curve_tools import rasterize


    class ConsecutiveRamped(RasterStyle):
        def __init__(self, pitch: DimFloat, ramp_start: float, ramp_end: float):
            """
            Raster style with ramped dwell times.
            The first spot has dwell time ``mill.dwell_time * ramp_start`` and the last
            ``mill.dwell_time * ramp_end``. All others are linearly interpolated.


            Args:
                pitch (DimFloat): pitch of spots, e.g. ``1 * unit('nm')``
                ramp_start (float): ramp start
                ramp_end (float): ramp end
            """
            if not has_length_dim(pitch):
                raise ValueError('pitch must have dimension [length].')
            self._pitch = pitch

            if ramp_start < 0 or ramp_end < 0:
                raise ValueError('ramp_start and ramp_end must not be negative.')

            self._ramp_start = ramp_start
            self._ramp_end = ramp_end

Next, the base property :attr:`~fibomat.raster_styles.RasterStyle.dimension` is implemented.
This method returns an integer indicating the dimensionality of the shapes which can be rasterized by this class. ::

    @property
    def dimension(self) -> int:
        return 1

Here, the dimensionality is constant one.

Finally, the actual rasterization method (:meth:`~fibomat.raster_styles.RasterStyle.rasterize`) must be provided.
See the documented code below for details. ::

    def rasterize(
        self,
        dim_shape: DimShape,
        mill: Mill,
        out_length_unit: LengthUnit,
        out_time_unit: TimeUnit
    ) -> RasterizedPattern:
        # Rasterize the passed shape with the pitch provided by the user.
        # The pitch must be scaled to the shape unit first to be consistent.
        points = np.array(rasterize(dim_shape.shape, scale_to(dim_shape.unit, self._pitch)).dwell_points)

        # Assign the dwell ramp to the weights of points
        points[:, 2] = np.linspace(self._ramp_start, self._ramp_end, len(points))

        # Scale the mill.dwell_time to the output time unit and multiply it to the dwell ramp values.
        points[:, 2] *= scale_to(out_time_unit, mill.dwell_time)

        # Scale the spots to the output length unit
        points[:, :2] *= scale_factor(out_length_unit, dim_shape.unit)

        # Create a RasterizedPattern object and return it.
        # np.tile repeats the points mill.repeats times.
        return RasterizedPattern(
            np.tile(points, (mill.repeats, 1)),
            length_unit=out_length_unit,
            time_unit=out_time_unit
        )

Now, the new raster style can be used ::

    from fibomat.layout import Layout
    from fibomat.linalg import Vector
    from fibomat.units import unit
    from fibomat import shapes, default_backends

    layout = Layout()
    site = layout.create_site(
        dim_position=Vector(0, 0) * unit('µm'),
        dim_fov=Vector(1, 1) * unit('µm')
    )
    site.create_pattern(
        dim_shape=shapes.Line((-.75, -.75), (.75, .75)) * unit('µm'),
        mill=Mill(dwell_time=1 * unit('ms'), repeats=5),
        raster_style=ConsecutiveRamped(pitch=1 * unit('nm'), ramp_start=1, ramp_end=np.pi)
    )
    layout.export(default_backends.SpotListBackend).save('rasterized.txt')

The complete source code can be found at `<https://github.com/fib-o-mat/fib-o-mat/blob/main/examples/custom_rasterization_style.py>`__.

