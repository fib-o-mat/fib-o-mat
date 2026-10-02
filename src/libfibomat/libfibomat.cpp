#include <algorithm>
#include <iostream>

// #include <range/v3/all.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include "arc_spline/arc_spline.h"
#include "arc_spline/arc_spline_tools.h"

// #include "svg_import/svg_import.h"

#include "utils.h"


typedef fibomat::arc_spline<double> arc_spline_t;


PYBIND11_MODULE(_libfibomat, m) {
    m.doc() = "C++ extension for fib-o-mat"; // optional module docstring
    m.attr("__name__") = "fibomat._libfibomat";

    // arc_spline_related
    py::class_<arc_spline_t>(m, "ArcSpline")
        // .def(py::init())
        .def(py::init<arc_spline_t const &>())
        .def(py::init([](py::array_t<double> curve_points, bool is_closed)
        {
            return arc_spline_t(fibomat::py_array_to_points(std::move(curve_points)), is_closed);
        }))
        .def("clone", &arc_spline_t::clone)
        .def_property_readonly("is_closed", &arc_spline_t::is_closed)
        .def_property_readonly("bounding_box", &arc_spline_t::bounding_box)
        .def_property_readonly("center", &arc_spline_t::center)
        .def("impl_translate", [](arc_spline_t& self, py::iterable iterable)
        {
            self.translate(fibomat::iterable_to_vector<double>(iterable));
        })
        .def("impl_rotate", [](arc_spline_t& self, double angle)
        {
            self.rotate(angle);
        })
        .def("impl_scale", [](arc_spline_t& self, double fac)
        {
            self.scale(fac);
        })
        .def("impl_mirror", [](arc_spline_t& self, py::iterable iterable)
        {
            self.mirror(fibomat::iterable_to_vector<double>(iterable));
        })
        .def_property_readonly("size",  [](arc_spline_t& self)
        {
            return self.size();
        })

        .def_property_readonly("orientation", &arc_spline_t::orientation)
        .def_property_readonly("length", &arc_spline_t::length)
        .def_property_readonly("area", &arc_spline_t::area)
        .def_property_readonly("start", &arc_spline_t::start)
        .def_property_readonly("end", &arc_spline_t::end)
        .def_property_readonly("vertices", &arc_spline_t::vertices)
        // .def("add_vertex", py::overload_cast<double, double, double>(&arc_spline_t::add_vertex))
        // .def("add_vertices", [](arc_spline_t& self, py::array_t<double> curve_points)
//        {
//            self.add_vertices(fibomat::py_array_to_points(std::move(curve_points)));
//        })
        .def("reverse", &arc_spline_t::reverse)
        // .def("translate", &arc_spline_t::translate)
        // .def("simple_scale", &arc_spline_t::simple_scale)
        // .def("simple_rotate", &arc_spline_t::simple_rotate)
        .def("visit", &arc_spline_t::visit<const std::function<bool(std::size_t, std::tuple<double, double, double>, std::tuple<double, double, double>)>& >)
        .def("contains", &arc_spline_t::contains)
        .def("closest_point", &arc_spline_t::closest_point)
        .def(py::pickle(
            [](const arc_spline_t &c) { // __getstate__
                return py::make_tuple(c.vertices(), c.is_closed());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");

                arc_spline_t c(t[0].cast<std::vector<std::tuple<double, double, double>>>(), t[1].cast<bool>());

                return c;
            }
        ))
    ;

    m.def("self_intersections", &fibomat::self_intersection<double>);
    m.def("curve_intersections", &fibomat::curve_intersections<double>);
    m.def("combine_curves", &fibomat::combine_curves<double>);
    m.def("offset_curve", &fibomat::offset_curve<double>);
    m.def("offset_with_islands", &fibomat::offset_with_islands<double>);
    m.def("convert_arcs_to_lines", &fibomat::convert_arcs_to_lines<double>);

    // svg_import
    // m.def("load_svg_from_file", &fibomat::load_svg_from_file);
    // m.def("load_svg_from_str", &fibomat::load_svg_from_str);


}
