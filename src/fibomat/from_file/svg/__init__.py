from typing import List

import svgelements

from fibomat.shapes import Shape
from fibomat.utils import PathLike


from typing import List, Optional, Callable
import warnings
import itertools

import numpy as np

try:
    import svgelements
except ModuleNotFoundError as error:
    raise RuntimeError(
        "You need to install the package with 'io' as optional dependency to be able to use the svg import."
    ) from error

# import anytree

from fibomat.units import Q_, U_
from fibomat.shapes import Shape, Rect, Circle, Polyline, Polygon, Polyline, Line, Ellipse
from fibomat.arrangements import Group
from fibomat.linalg import mirror, translate, Vector, BoundingBox
from fibomat.utils import PathLike

# def path_type(path: svgelements.Path):
#     for segment in path:
#         if not isinstance(segment, svgelements.MO):

class IndexedSubpath:
    def __init__(self, subpath, index):
        self.subpath = subpath
        self.index = index


# def make_bbox(path: IndexedSubpath) -> BoundingBox:
#     # box: xmin, ymin, xmax, ymax
#     box = path.subpath.bbox()
#     print(box)
#     return BoundingBox(box[:2], box[2:])



# def buildpath_tree_impl(paths: List[IndexedSubpath], _root_node, root_bbox, i_node):
#     path = paths.pop(0)
#     path_bbox = make_bbox(path)


#     if path_bbox in root_bbox:
#         pass
#     pass 


# def sort_by_bboxes(paths: List[IndexedSubpath]):
#     def _get_max_bbox(paths_):


#     bboxes = [make_bbox(path) for path in paths]

#     i_sorted = []



# def build_path_tree(paths: List[IndexedSubpath]):
#     # we assume that the paths are hierarchically ordered (meaning that a subpath which lies completely in a nother one comes after the path it lies in) 

#     class PathData:
#         def __init__(self, /, **kwargs):
#             self.__dict__.update(**kwargs)

#     if not paths:
#         return

#     root = anytree.Node('root')

#     nodes = []
#     for i, path in enumerate(paths):
#         nodes.append(anytree.Node(i, parent=root, path_data=PathData(path=path.subpath, bbox=make_bbox(path), index=path.index)))

#     nodes.sort(key=lambda node: -node.path_data.bbox.area)

#     print([node.path_data.bbox for node in nodes])
    
#     def build_subtree(root_node, nodes):
#         while nodes:
#             possible_child = nodes.pop(0)
#             if possible_child.path_data.bbox in root_node.path_data.bbox:
#                 possible_child.parent = root_node
#             else:
#                 possible_child.parent = root_node.parent
#                 build_subtree(possible_child, nodes)
    
#     # print(current_root)

#     build_subtree(nodes[0], nodes[1:])

#     # while nodes:
#     #     current_root = nodes[0]
#     #     print('current root', current_root.name)

#     #     for possible_child in nodes[1:]:
#     #         if possible_child.path_data.bbox in current_root.path_data.bbox:


#     # # for _, _, node in anytree.RenderTree(root): 
#     # for node in nodes[1:]:
#     #     print('current root', current_root.name)
#     #     print('node', node.name, node.path_data)
#     #     print(node.path_data.bbox in current_root.path_data.bbox)

#     #     if node.path_data.bbox in current_root.path_data.bbox:
#     #         print(f'node {node.name} ind {current_root.name}')
#     #         node.parent = current_root
#     #         current_root = node
#     #     else:
#     #         # current_root = current_root.parent
#     #         print('searching new root, old root is ', current_root.name)
#     #         while node.path_data.bbox in current_root.path_data.bbox and current_root.parent.name != 'root':
#     #             current_root = current_root.parent

#     #             print('new root', current_root)
    
#     for pre, fill, node in anytree.RenderTree(root):
#         print("%s%s" % (pre, node.name))
#     # current_root_path = paths.pop(0)

#     # anytree.Node(i_node, parent=_root, segment=current_root_path)
#     # i_node += 1


#     # current_root_bbox = make_bbox(current_root_path)

    
#     #     if make_bbox(path) in current_root_bbox:
#     #         pass

#     # return _root

#     return None


def path_is_closed(path: svgelements.Subpath):
    segments = path.segments()
    start = segments[0].end
    end = segments[-1].end

    return np.allclose(start, end)


def is_polyline_or_polygon(path: svgelements.Subpath):
    polyline_segment_types = svgelements.Move, svgelements.Line, svgelements.Close
    for segment in path.segments():
        if not isinstance(segment, polyline_segment_types):
            return False
    
    return True


def categorize(elements: List[IndexedSubpath], condition: Callable):
    first, second = [], []
    for element in elements:
        (first, second)[not condition(element.subpath)].append(element)

    return first, second
    

def parse_polyline(polyline_subpaths: IndexedSubpath):
    points = []

    segments = list(polyline_subpaths.subpath.segments())

    for segment in segments[1:]:
        start = segment.start
        # end = segment.end
        # print(start)
        points.append(Vector(*start))
    
    points.append(Vector(*segments[-1].end))

    return IndexedSubpath(Polyline(points), polyline_subpaths.index)


def parse_polygon(polygon_subpaths: IndexedSubpath):
    points = []

    segments = list(polygon_subpaths.subpath.segments())

    for segment in segments[1:]:
        start = segment.start
        # end = segment.end
        # print(start)
        points.append(Vector(*start))

    print(segments)

    return IndexedSubpath(Polygon(points), polygon_subpaths.index)


def parse_path(path: svgelements.Path):
    # https://stackoverflow.com/a/12135169

    subpaths = [IndexedSubpath(subpath, i) for i, subpath in enumerate(path.as_subpaths())]

    closed_paths, open_paths = categorize(subpaths, path_is_closed)
    

    polylines, non_polyline_open_paths = categorize(open_paths, is_polyline_or_polygon)

    polygons, non_polygon_closed_paths = categorize(closed_paths, is_polyline_or_polygon)

    if non_polyline_open_paths:
        warn_element_not_supported('non_polyline_open_paths')

    if non_polygon_closed_paths:
        warn_element_not_supported('non_polygon_closed_paths')

    fibomat_polygons = [parse_polygon(polygon) for polygon in polygons]

    fibomat_polylines = [parse_polyline(polyline) for polyline in polylines]
    # build_path_tree(polygons)
    
    
    # [ for polygon in polygons]

    # print(polylines)
    # print(open_curves)
    # print(closed_paths)

    to_be_chained = []

    if fibomat_polygons:
        print(fibomat_polygons)

        to_be_chained.append(fibomat_polygons)

    if fibomat_polylines:
        print(fibomat_polylines)
        to_be_chained.append(fibomat_polylines)

    merged = list(itertools.chain(
        *to_be_chained
        # open_curves
    ))

    print('merged', merged)

    if merged:
        return Group(list(map(
            lambda index_path: index_path.subpath,
            sorted(merged, key=lambda index_path: index_path.index)
        )))

    # raise RuntimeError



def is_reified(element: svgelements.SVGElement) -> bool:
    def matrix_as_array(m):
        return np.array([m.a, m.c, m.b, m.d, m.e, m.f], dtype=np.float64)
    unit_matrix = np.array([1., 0., 0., 1., 0., 0.], dtype=np.float64)

    return np.allclose(matrix_as_array(element.transform), unit_matrix)


def warn_element_not_supported(element: svgelements.SVGElement, reason: Optional[str] = None):
    msg = f'Unsupported shape of type "{type(element)}"'
    if reason:
        msg += f' (reason: {reason})'
    
    msg += '. Try to convert it to path in your favorite svg editor first. For now, it will be skipped.'
    
    warnings.warn(msg)


def parse_elements(elements: List[svgelements.SVGElement]) -> List:
    parsed_shapes = []

    for element in elements:
        try:
            if element.values['visibility'] == 'hidden':
                continue
        except (KeyError, AttributeError):
            pass
        
        if isinstance(element, svgelements.SVG):
            continue
        elif isinstance(element, svgelements.Path):
            if len(element) != 0:
                reified_path = element.reify()
                if paths_group := parse_path(reified_path):
                    parsed_shapes.append(paths_group)
        elif isinstance(element, svgelements.Shape):
            reified_shape = element.reify()

            if is_reified(reified_shape):
                if isinstance(reified_shape, svgelements.Rect):
                    rect: svgelements.Rect = reified_shape
                    if reified_shape.rx > 0 or reified_shape.ry > 0:
                        warn_element_not_supported(reified_shape, 'rounded corners')
                    else:
                        parsed_shapes.append(Rect(
                            width=rect.width, height=rect.height, center=(rect.x + rect.width / 2, rect.y + rect.height / 2)
                        ))
                elif isinstance(reified_shape, svgelements.Circle):
                    circle: svgelements.Circle = reified_shape
                    parsed_shapes.append(Circle(
                            r=circle.r, center=(circle.cx, circle.cy)
                    ))
                elif isinstance(reified_shape, svgelements.Ellipse):
                    ellipse: svgelements.Ellipse = reified_shape
                    parsed_shapes.append(Ellipse(
                            a=ellipse.rx, b=ellipse.ry, center=(ellipse.cx, ellipse.cy)
                    ))
                elif isinstance(reified_shape, svgelements.Line):
                    line: svgelements.Line = reified_shape
                    parsed_shapes.append(Line(
                            start=line.start, end=line.end
                    ))
                elif isinstance(reified_shape, (svgelements.Polyline, svgelements.Polygon)):
                    parsed_shapes.append(parse_path(reified_shape))
                else:
                    warn_element_not_supported(reified_shape)
            else:
                # path can be always reified
                reified_path = svgelements.Path(element).reify()
                if paths_group := parse_path(reified_path):
                    parsed_shapes.append(paths_group)
        elif isinstance(element, svgelements.Group):
            if group_elements := parse_elements(element):
                parsed_shapes.append(
                    Group(group_elements)
                )
        else:
            # raise RuntimeError(f'Unsupported shape "{element}".')
            warn_element_not_supported(element)

    return parsed_shapes


def shapes_from_svg(file_path: PathLike, scale: Q_) -> Optional[Group]:
    """[summary]

    Args:
        file_path (PathLike): [description]
        scale (Q_): defines the length reference. E.g.: if scale = 1cm, 1cm in the svg file will be 1 length unit in the import shapes.

    Returns:
        Group: [description]
    """

    scale = scale.m_as('inch')

    svg = svgelements.SVG.parse(file_path, ppi=1/scale)

    if parsed_svg := parse_elements(svg):
        return Group(parsed_svg).transformed(mirror((1, 0)) | translate((-svg.width / 2, svg.height / 2)))