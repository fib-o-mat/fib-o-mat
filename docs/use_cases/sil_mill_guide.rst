=====================================================================
Using the SILMill
=====================================================================

The ``SILMill`` (Solid-Immersion-Lens-Mill) is a specialized mill that can be used to easily mill solid immersion lenses, since the necessary dwell time function is readily implemented. 

Basic Example
-------------

Here is a simple example of creating a SIL with the ``SILMill``:

.. code-block:: python

    from fibomat import Sample, U_, Q_, SILMill
    from fibomat import shapes, raster_styles
    from fibomat.default_backends.fei import FEIStreamFile

    # Create a sample
    s = Sample()

    # Create a site with 20 µm field of view
    site = s.create_site(
        dim_position=(0, 0) * U_('µm'),
        dim_fov=(20, 20) * U_('µm')
    )

    # Define a SILMill with sphere radius 3.953 µm and total radius 7.370 µm
    sil_mill = SILMill(
        radius_sil=3953 * U_('nm'),  # Convert to quantity
        radius=7370 * U_('nm'),
        repeats=1,
        min_dwell_time=0.1,  # µs
        max_dwell_time=10    # µs
    )

    # Define a raster style
    spiral_style = raster_styles.two_d.Spiral(
        pitch=20 * U_('nm'),
        spiral_pitch=20 * U_('nm'),
        scan_sequence=raster_styles.ScanSequence.CONSECUTIVE,
        direction="out-in"
    )

    # Create a circular shape to mill
    circle = shapes.Circle(r=7370, center=(0, 0))

    # Create a pattern combining the shape, mill, and raster style
    site.create_pattern(
        dim_shape=circle * U_('nm'),
        mill=sil_mill,
        raster_style=spiral_style
    )

    # Plot the site to verify the correct shape
    s.plot(rasterize_pitch=Q_('0.01 µm'), plot_rasterized=True)

    # Export to FEI stream file
    exported = s.export(FEIStreamFile, n_rep=684, margin=0.76)
    exported.save('sil_pattern.str')



Key Parameters
--------------

**radius_sil** (QuantityType)
    Radius of the spherical part of the profile.

**radius** (QuantityType)
    Total radius including the conical section. Points between ``radius_sil`` and ``radius`` follow a linear falloff. Points beyond ``radius`` are not milled.

**repeats** (int, optional)
    Number of times the pattern is repeated during patterning. Normally, you want to leave this at 1.

**min_dwell_time** (float, optional)
    Minimum dwell time (in microseconds) to avoid errors thrown by the microscope. Insert the minimal possible dwelltime your microscope accepts here.

**max_dwell_time** (float, optional)
    Maximum dwell time (in microseconds) at the center of the sphere. Default: 10 µs. Be careful about raising this parameter, prefer to repeat the stream file more often.





Resulting Profile
-------------------------

For a point at distance *r* from the center, the dwell time will be:

- If r < ``radius_sil``: 
  
  .. math::
     t(r) = \text{max\_dwell\_time} - \frac{\text{max\_dwell\_time}}{\text{radius\_sil}} \sqrt{\text{radius\_sil}^2 - r^2}

- If ``radius_sil`` ≤ r < ``radius``:
  
  .. math::
     t(r) = \text{max\_dwell\_time} \left(1 - \frac{r - \text{radius\_sil}}{\text{radius} - \text{radius\_sil}}\right)

- If r ≥ ``radius``:
  
  .. math::
     t(r) = 0



