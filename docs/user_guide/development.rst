Development
===========

|:test_tube:|  Contributing
---------------------------
To contribute custom code, follow the steps below.

    1. fork fib-o-mat
    2. create a new branch, e.g. ``git checkout -b my-new-branch``
    3. commit your changes, ``git add ...``, ``git commit -m "..."``
    4. push the code. ``git push origin my-new-branch``
    5. create a pull request from the fork (see `here <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`__)

|:test_tube:| Versioning
------------------------
Versioning is done with help of `bump2version <https://github.com/c4urself/bump2version>`__.
Run

.. code-block:: bash

    $ bump2version {major|minor|patch}

in the root folder of fib-o-mat to increase the corresponding number. Push the resulting commit to the git repository.

|:test_tube:| Building the docs
-------------------------------

Install the package with the extra ``docs`` (this also installs the Sphinx extensions and the packages which are needed to create the plots)

.. code-block:: bash

    $ pip install -e ".[docs]"

and run

.. code-block:: bash

    $ make html

in the ``[fib-o-mat]/docs`` folder. The documentation is written to ``[fib-o-mat]/build/sphinx/html``.

The documentation consists of the pages in the ``docs`` folder, the API reference, which is generated from the docstrings of the code, and the changelog (``CHANGELOG.md`` in the root folder). The plots in the documentation are created by running the scripts in the ``examples`` folder (see the directive ``fibomat-plot`` in ``docs/_ext/fibomat_plot.py``); they are interactive and need no internet connection.

The documentation is published on `Read the Docs <https://fib-o-mat.readthedocs.io/>`__. The version selector in the navigation bar uses ``docs/_static/switcher.json``; add an entry for each new release to this file.

|:test_tube:| Building wheel packages (for PyPI)
-------------------------------------------------

The package is built with `maturin <https://www.maturin.rs>`__. The extension module uses the stable ABI of Python 3.8 (``abi3``), so one wheel per platform covers all Python versions from 3.8 on. Install maturin with ``pip install maturin``.

Source distribution
+++++++++++++++++++

.. code-block:: bash

    $ maturin sdist

The source distribution is placed in ``[fib-o-mat]/target/wheels``.

Linux
+++++

Wheels for Linux are built in a ``manylinux`` container, e.g. with the container of maturin (`docker <https://www.docker.com/>`__ must be installed):

.. code-block:: bash

    $ docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release

The wheels are placed in ``[fib-o-mat]/target/wheels``.

Windows
+++++++

Run

.. code-block:: bash

    $ maturin build --release

on a MS Windows system with the Rust toolchain and the Visual Studio build tools installed. The wheel is placed in ``[fib-o-mat]\target\wheels``.
