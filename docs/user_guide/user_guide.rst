User guide
==========

The user guide introduces all features of the fib-o-mat package.

In the following, the basic concepts of the library are summarized. The focus is on the programming part. For more background information, see the associated publication which is linked at the :ref:`starting page <index:fib-o-mat>`.

In short, fib-o-mat is a Python library to generate patterns for ion beam instruments. This is implemented as a two-step process. First, a patterning shape must be defined. In a second step, the shape is equipped with beam and rasterization settings. The final pattern design can be exported to a microscope-readable format.

During pattern creation, two different paths can be taken. The first one is called the 'high-level approach'. This means that the shape and the rasterization settings are defined and fib-o-mat does the rasterization automatically.

Alternatively, the shapes can be rasterized by hand. This allows the user to have very fine control over the pattern design and to apply optimizations to the rasterized points, for example ('low-level approach').

Of course, both approaches can be combined.

All sections marked with a test tube |:test_tube:| describe advanced features of the package which may be skipped on first usage.

.. warning:: Currently, fib-o-mat only contains exporting backends for a few microscope software packages. Support for other systems must be added by the user and is explained :ref:`here <user_guide/extending:extending fib-o-mat>`. Even so, the provided backends can easily be modified to support common microscopes.

.. warning:: Some parts of the user guide are still missing and will be added soon.

.. toctree::
    :maxdepth: 2
    :hidden:

    Installation <installation>
    Preliminary <preliminary>
    General structure <general_structure>
    Geometric shapes <geometric-shapes>
    Mill & rasterization settings <mill_rasterizing>
    Exporting & visualization <exporting_visualization>
    Grouping & arranging <grouping_arranging>
    Extending fib-o-mat <extending>
    Development <development>
