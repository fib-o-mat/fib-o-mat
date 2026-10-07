Installation
============

fib-o-mat requires Python 3.8 or newer and can be installed via pip on most systems. It contains an extension module which is written in Rust; pre-built packages (wheels) are available for 64-bit Linux and Windows systems. It is highly recommended to use a virtual environment, e.g.

.. code-block:: bash

    $ python -m venv .venv
    # for *nix systems
    $ source .venv/bin/activate
    # for MS Windows
    $ .venv\Scripts\activate.bat
    # install fib-o-mat
    $ pip install --upgrade fibomat

See `the Python documentation <https://docs.python.org/3/library/venv.html>`__ for more information on virtual environments.

Optional dependencies
---------------------

Some features need additional packages. They can be installed with the extras of the package, e.g. ``pip install "fibomat[exporting,io]"``:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Extra
      - Content
    * - ``exporting``
      - plotting and exporting backends (bokeh, pillow, xmltodict, marshmallow, ...)
    * - ``io``
      - import of vector graphics (SVG and DXF files)
    * - ``gui``
      - the ion beam simulation (``beam_simulation``)
    * - ``experimental``
      - features which are not stable or not fully reworked yet, e.g. the ``ContourParallel`` raster style (numba)
    * - ``docs``
      - building this documentation
    * - ``testing``
      - running the tests
    * - ``dev``
      - tools to develop fib-o-mat

Building from source
--------------------

If no suitable pre-built package is found, the pip command above compiles the package. This requires a `Rust toolchain <https://www.rust-lang.org/tools/install>`__; the build is carried out by `maturin <https://www.maturin.rs>`__ (which pip installs automatically).

.. note:: macOS is currently not officially supported. Even so, the package might be built correctly and can probably be used without problems.

Clone the git repository with

.. code-block:: bash

    $ git clone https://github.com/fib-o-mat/fib-o-mat

and run one of the following commands in the fib-o-mat directory

.. code-block:: bash

    $ pip install .
    # or
    $ pip install -e .

The latter installs fib-o-mat in development mode (the Python files are used from the source folder). In development mode, the extension module is built in the virtual environment with

.. code-block:: bash

    $ pip install maturin
    $ maturin develop --release
