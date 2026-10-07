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

|:test_tube:| Releasing
-----------------------

Versioning is done with the help of `bump2version <https://github.com/c4urself/bump2version>`__ (configured in ``.bumpversion.cfg``).
To publish a new release, follow these steps in the root folder of fib-o-mat. ``[version]`` is the new version number, e.g. ``0.7.0``.

    1. Make sure that all tests pass (``pytest tests/``) and that the documentation builds without warnings (see "Building the docs" below).
    2. Check that ``CHANGELOG.md`` describes all changes in the ``## [Unreleased]`` section. Breaking changes must be marked with **Breaking:** (and summarized at the top for larger releases).
    3. Add the new version to ``docs/_static/switcher.json`` (below the entries ``latest`` and ``stable``, newest version first). The ``version`` value must match the name of the version on Read the Docs, which is the git tag ``v[version]``.
    4. Commit these changes. ``bump2version`` refuses to run in a dirty working directory.
    5. Run

       .. code-block:: bash

           $ bump2version {major|minor|patch}

       This increases the version in ``pyproject.toml`` and ``src/fibomat/__init__.py``, renames the ``[Unreleased]`` heading in ``CHANGELOG.md`` to ``[[version]] - [current date]``, creates a commit and tags it with ``v[version]``.
    6. Add a new, empty ``## [Unreleased]`` heading above the new release at the top of ``CHANGELOG.md`` (``bump2version`` cannot insert it) and commit.
    7. Push the commits and the tag: ``git push && git push --tags``.
    8. Build the packages (see "Building wheel packages" below) and upload them with ``twine upload target/wheels/*``.
    9. Activate the new version ``v[version]`` on `Read the Docs <https://readthedocs.org/projects/fib-o-mat/>`__ (if it is not activated automatically) and check that it appears in the version selector of the documentation.

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
