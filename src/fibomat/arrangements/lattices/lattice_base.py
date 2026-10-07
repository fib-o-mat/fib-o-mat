from typing import Iterable, Union, Callable, Optional, Tuple, List, Type
import abc

import numpy as np

from fibomat.composite_shapes import HollowArcSpline
from fibomat.shapes import ArcSplineCompatible, ArcSpline, Line
from fibomat.linalg import Transformable, Vector, DimVector
from fibomat.curve_tools import fill_with_lines
from fibomat.arrangements.arrangementbase import ArrangementBase
from fibomat.arrangements.utils import _round_towards_mean, _check_lattice_vectors
from fibomat.arrangements.groups.group_base import GroupBase
from fibomat.linalg.helpers import GeomLine


class LatticeBaseMixin(abc.ABC):
    _VectorType: Type[Union[Vector, DimVector]]

    @staticmethod
    def _gen_lattice_points(
        boundary: Union[HollowArcSpline, ArcSpline], u: Vector, v: Vector, seed: Vector
    ) -> Tuple[np.ndarray, np.ndarray]:
        # boundary = boundary.translated_to(center)

        alpha = u.angle_about_x_axis

        # map alpha to [-pi/2, pi/2]
        if alpha < np.pi/2:
            pass
        elif np.pi/2 <= alpha < 1.5*np.pi:
            alpha -= np.pi
        else:
            alpha -= 2*np.pi

        pitch = (v - u.projected(v)).length

        lattice_planes = fill_with_lines(boundary, pitch=pitch, alpha=alpha, invert=False, seed=seed)

        plane_normal = GeomLine(v, seed)
        # print(boundary, lattice_planes, seed)



        # from fibomat.layout import Layout
        # from fibomat import U_
        # from fibomat.shapes import Spot
        # s = Layout()
        # s.add_annotation(boundary * U_('µm'))
        # for plane in lattice_planes:
        #     for line in plane:
        #         s.add_annotation(line * U_('µm'))

        # s.add_annotation(Spot(seed) * U_('µm'), color='red')

        # s.add_annotation(Line(seed, seed + u) * U_('µm'), color='yellow')
        # s.add_annotation(Line(seed, seed + v) * U_('µm'), color='orange')

        # s.plot()


        class NoLatticePointsOnLine(Exception):
            pass
        
        # s.plot()    

        def lattice_points_on_line(line: Line):
            m = np.array([u, v]).T

            start_lattice_point = np.linalg.solve(m, line.start + seed)
            end_lattice_point = np.linalg.solve(m, line.end + seed)


            intersection = plane_normal.intersect_at_param(GeomLine(u, line.start))

            i_v = np.sign(intersection) * int(np.round(Vector(plane_normal(intersection) - seed).mag / v.mag))

            plane_line = GeomLine(u, i_v * v + seed)

            # s.add_annotation(Line(i_v * v + seed, i_v * v + seed + u) * U_('µm'), color='pink')
            # s.plot()
            # print(line.start)

            start_lattice_point = plane_line.find_param(line.start), i_v
            end_lattice_point = plane_line.find_param(line.end), i_v

            # s.add_annotation(Spot(seed + v * i_v) * U_('µm'))

            # raise RuntimeError

            # s.add_annotation(Spot(start_lattice_point[0] * u + start_lattice_point[1] * v + seed) * U_('µm'), color='blue')
            # s.add_annotation(Spot(end_lattice_point[0] * u + end_lattice_point[1] * v + seed) * U_('µm'), color='green')

            i_u_start, i_u_end = _round_towards_mean(start_lattice_point[0], end_lattice_point[0])

            # TODO: more tests on this!
            if abs(start_lattice_point[0] - end_lattice_point[0]) < 1:
                if not np.isclose(i_u_start, i_u_end):
                    # line does not contain any lattice point
                    raise NoLatticePointsOnLine

            # TODO: needed?
            # assert np.isclose(start_lattice_point[1], end_lattice_point[1])
            # print(start_lattice_point[1], np.rint(start_lattice_point[1]))
            # assert np.isclose(start_lattice_point[1], np.rint(start_lattice_point[1]))

            if i_u_start <= i_u_end:
                i_u = np.arange(i_u_start, i_u_end + 1, dtype=int)
            else:
                i_u = np.arange(i_u_end, i_u_start + 1, dtype=int)[::-1]
            # i_v = int(np.rint(start_lattice_point[1]))

            return i_u, i_v

        i_u_min = np.iinfo(int).max
        i_u_max = np.iinfo(int).min

        i_v_min = np.iinfo(int).max
        i_v_max = np.iinfo(int).min

        lattice_points = []

        for plane in lattice_planes: # reversed(lattice_planes):
            for line in plane:
                # s.add_annotation(line * U_('µm'))
                try:
                    points = lattice_points_on_line(line)

                except NoLatticePointsOnLine:
                    continue
                i_points_u, i_point_v = points

                i_u_min = min(i_u_min, np.min(i_points_u))
                i_u_max = max(i_u_max, np.max(i_points_u))

                i_v_min = min(i_v_min, i_point_v)
                i_v_max = max(i_v_max, i_point_v)

                lattice_points.append(np.c_[i_points_u, [i_point_v]*len(i_points_u)])

                # for p in lattice_points[-1]:
                #     s.add_annotation(Spot(p[0] * u + p[1] * v  + seed) * U_('µm'), color='yellow')
                # s.plot()

        lattice_points_uv = np.concatenate(lattice_points)
        lattice_points_xy = np.outer(lattice_points_uv[:, 0], u) + np.outer(lattice_points_uv[:, 1], v) + seed

        return lattice_points_uv, lattice_points_xy

    @classmethod
    def _generate_impl(
        cls,
        boundary: Union[HollowArcSpline, ArcSplineCompatible],
        u: Vector, v: Vector,
        center: Vector,
        element_gen: Callable,
        predicate: Optional[Union[Callable, List[Callable]]],
        explode: bool, remove_outliers: bool,
        scale_vec: Callable,
        unscale_vec: Callable,
        seed: Vector
    ):
        # check arguments
        if not isinstance(boundary, HollowArcSpline) and not isinstance(boundary, ArcSplineCompatible):
            raise TypeError('boundary must be HollowArcSpline or ArcSplineCompatible')

        if isinstance(boundary, ArcSplineCompatible):
            boundary = boundary.to_arc_spline()

        if not boundary.is_closed:
            raise ValueError('boundary must be closed shaped.')

        _check_lattice_vectors(u, v)

        if predicate is not None:
            if not isinstance(predicate, Iterable):
                predicate = (predicate,)

            if not all([callable(pred) for pred in predicate]):
                raise TypeError('predicate must be Callable or List[Callable].')

        explode = bool(explode)
        remove_outliers = bool(remove_outliers)

        # generate actual lattice site objects
        lattice_points_uv, lattice_points_xy = cls._gen_lattice_points(boundary, u, v, seed)

        # shift uv points so that (0, 0) is in the upper left

        # min_u, min_v = np.min(lattice_points_uv, axis=0)

        lattice_points_uv -= np.min(lattice_points_uv, axis=0)
        lattice_points_uv = lattice_points_uv.astype(int)

        assert np.min(lattice_points_uv) >= 0

        elements_by_uv = np.empty(shape=(np.max(lattice_points_uv[:, 1] + 1), np.max(np.max(lattice_points_uv[:, 0] + 1))), dtype=object)

        elements = []
        for lattice_point_xy, lattice_point_uv in zip(lattice_points_xy, lattice_points_uv):
            lattice_site_element = element_gen(unscale_vec(lattice_point_xy), lattice_point_uv)

            u, v = lattice_point_uv

            elements_by_uv[v, u] = []

            if lattice_site_element:
                lattice_site_element = lattice_site_element.translated_to(unscale_vec(lattice_point_xy + center))

                if explode and isinstance(lattice_site_element, ArrangementBase):
                    sub_elements = lattice_site_element._arrangement_elements()
                else:
                    sub_elements = [lattice_site_element]
                        
                if remove_outliers:
                    for sub_elem in sub_elements:
                        # TODO: allow user to select different methods
                        # for now, bounding box is used
                        add_shape = True
                        for corner in sub_elem.bounding_box.corners:
                            if not boundary.contains(scale_vec(corner)):
                                add_shape = False
                                break
                        if add_shape:
                            elements.append(sub_elem)
                            elements_by_uv[v, u].append(sub_elem)
                                # if boundary.contains(scale_vec(sub_elem.pivot)):
                                #     elements.append(sub_elem)
                                #     elements_by_uv[v, u].append(sub_elem)


                                    # if elements_by_uv[v, u]
                else:
                    elements.extend(sub_elements)
                    elements_by_uv[v, u].extend(sub_elements)



        # sort the points by predicate
        if predicate:
            if explode:
                lattice_elements_xy = np.array([elem.pivot for elem in elements])
            else:
                lattice_elements_xy = lattice_points_xy

            # todo: this looks weird
            sorted_indices = np.lexsort(tuple(pred(lattice_elements_xy) for pred in predicate))

            elements = [elements[i] for i in sorted_indices]

        return elements, elements_by_uv
