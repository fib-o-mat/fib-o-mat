=========
fib-o-mat
=========

A Python toolbox to generate focused ion beam patterning layouts
----------------------------------------------------------------

.. figure:: /_static/flowchart.png
    :align: center

fib-o-mat is a Python library to create beam patterns for focused ion beam (FIB) instruments.

Features:
    * built-in modeling of patterning geometries
    * customizable rasterization styles
    * optimization of patterning geometries and rasterized patterns
    * extendable

Pattern geometries can be modeled directly in Python based on (pre-)defined geometric primitives or imported from vector graphics. They can be equipped with beam and rasterization settings and exported to microscope-compatible files.

fib-o-mat is designed to be flexible and easily expandable. Hence, adding support for different microscopes, custom geometric primitives or optimization routines is a straightforward process.

To use fib-o-mat, basic Python knowledge and a good understanding of the target microscope are required.

.. grid:: 1 2 2 4
    :gutter: 2

    .. grid-item-card:: Getting started
        :link: getting_started
        :link-type: doc

        Install fib-o-mat and create your first patterning layout.

    .. grid-item-card:: User guide
        :link: user_guide/user_guide
        :link-type: doc

        All features of the library with examples.

    .. grid-item-card:: API reference
        :link: api/index
        :link-type: doc

        The documentation of all classes and functions.

    .. grid-item-card:: Changelog
        :link: changelog
        :link-type: doc

        What changed in which version.

Please use the `issue system on GitHub <https://github.com/fib-o-mat/fib-o-mat/issues>`__ for bug reports and questions concerning the package.

Made with |:black_heart:| and |:coffee:| at `HZB <https://www.helmholtz-berlin.de/>`__ and `FBH <https://www.fbh-berlin.de/en/>`__ in Berlin.

If you use this library in your work, please cite

Deinhart, V., Kern, L.-M., Kirchhof, J. N., Juergensen, S., Sturm, J., Krauss, E., Feichtner, T., Kovalchuk, S., Schneider, M., Engel, D., Pfau, B., Hecht, B., Bolotin, K. I., Reich, S., & Höflich, K. (2021). The patterning toolbox FIB-o-mat: Exploiting the full potential of focused helium ions for nanofabrication. Beilstein Journal of Nanotechnology, 12(1), 304–318. https://doi.org/10.3762/bjnano.12.25


.. toctree::
    :maxdepth: 2
    :hidden:

    Getting started <getting_started>
    User guide <user_guide/user_guide>
    Use cases <use_cases/introduction>
    API reference <api/index>
    Changelog <changelog>
    Contributors <contributors>
    License <license>
