<!--- some hacky stuff get everything working on github and pypi --->
<div align="center">

<img src="https://fib-o-mat.readthedocs.io/en/latest/_static/fibomat.png" width="50%" style="width: 50%; display: block; margin-left: auto; margin-right: auto;"  alt="fibomat logo" />
</div>

fib-o-mat is a Python library to create beam patterns for focused ion beam instruments.

Pattern geometries can be modeled directly in Python based on (pre-)defined geometric primitives or imported from vector
graphics. These can be equipped with beam and rasterizing settings and exported to microscope compatible files.

fib-o-mat is by designed flexible and easily expandable. Hence, adding support for for different microscopes, custom
geometric primitives or optimization routines is a straightforward process.

For the usage of fib-o-mat, basic python knowledge and a good understanding of the target microscope are mandatory.
See the [getting started guide](https://fib-o-mat.readthedocs.io/en/latest/getting_started.html) for an introduction to
this library and the [user guide](https://fib-o-mat.readthedocs.io/en/latest/user_guide/user_guide.html) for a complete
documentation. The module reference is to be found [here](https://fib-o-mat.readthedocs.io/en/latest/reference.html).

<img src="https://fib-o-mat.readthedocs.io/en/latest/_images/flowchart.png" style="width: 100%;  height: auto;" alt="workflow" />

Made with :black_heart: and :coffee: at HZB and FBH in Berlin.

If you use this library in your work, please cite

Deinhart et al., Beilstein J. Nanotechnol. **2021**, 12, 304-318. [doi:10.3762/bjnano.12.25](https://www.beilstein-journals.org/bjnano/articles/12/25)


Installation
============
Run in a terminal
```
$ pip install fibomat
```  
It is highly recommended to use virtual environments.

There are several optional dependencies to install:

- `exporting`: plotting and exporting patterns
- `io`: loading SVG and DXF
- `gui`: beam simulation gui

E.g., install fib-o-mat alongside the `exporting` dependencies via
```
$ pip install fibomat[exporting]
```  

Example
=======
```python
from fibomat import Sample, Mill, Q_, U_
from fibomat.shapes import Line
from fibomat import raster_styles
from fibomat.default_backends import SpotListBackend

# create a Sample class object with optional description
sample = Sample('Useful description here')

# add a site to the sample with cente = (0, 0) and field of view of (10, 10)
site = sample.create_site(dim_position=([0, 0], U_('µm')), dim_fov=([10, 10], U_('µm')))  # '%*µm*)'

# create a Pattern with a Line shape and add it to the site
site.create_pattern(
    dim_shape=(Line((-5, 0), (5, 0)), U_('µm')),
    shape_mill=Mill(dwell_time=Q_('5 ms'), repeats=1),
    raster_style=raster_styles.one_d.Linear(pitch=Q_('1 nm'))
)

# export a rasterized version of the pattern as text file in a pre-defined (but editable) format. See docs for details.
sample.export(SpotListBackend).save('pattern.txt')

# plot the pattern
sample.plot()
```

License
=======

The source code is licensed under the GNU General Public License v3.0. This includes everything besides the 'docs' folder and its content in the git repository. See LICENSE.txt for a copy of the license.

The documentation is licensed under the Creative Commons Attribution 4.0 International. This includes everything in the 'docs' folder in the git repository and the documentation hosted at https://fib-o-mat.readthedocs.io/. A copy of the license is to be found at 'docs/LICENSE_DOCS.txt' in the git repository.
