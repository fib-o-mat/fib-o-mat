"""Module provides support for import of vector graphics."""
from typing import Optional
import pathlib

from fibomat.shapes import Shape

from fibomat.from_file.dxf import shapes_from_dxf
from fibomat.from_file.svg import shapes_from_svg
from fibomat.utils import PathLike
from fibomat.arrangements import Group
from fibomat.units import LengthQuantity


def shapes_from_file(file_path: PathLike, scale: LengthQuantity) -> Optional[Group]:
    """Parse a vector graphic file and return the contained data mapped to fib-o-mat shape types.

    The file type is determined by the file_path suffix. Currently supported is only dfx-format.

    Args:
        file_path (PathLike): file to be parsed
        scale (LengthQuantity): scale to be used for import.  E.g.: if scale = 1cm, 1cm in a svg file will be 1 length unit in fibomat.

    Returns:
        List[Shapes]

    Raises:
        RuntimeError: Raised if provided file is not valid.
        RuntimeError: Raised if provided file format is not supported.
    """
    file_path = pathlib.Path(file_path).absolute()

    if not file_path.is_file():
        raise RuntimeError(f'"{file_path}" is not a valid file')

    suffix = file_path.suffix

    if suffix == '.dxf':
        return Group(shapes_from_dxf(file_path))
    elif suffix == '.svg':
        return shapes_from_svg(file_path, scale)

    raise RuntimeError(f'Cannot import file with extension "{suffix}"')
