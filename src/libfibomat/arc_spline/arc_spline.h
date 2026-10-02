#ifndef FIB_O_MAT_CURVE_H
#define FIB_O_MAT_CURVE_H


#include <stdexcept>
#include <cmath>

#include "cavc/polyline.hpp"

namespace
{
    /**
     * @brief Convert a cavc::PlineVertex to a tuple (x, y, bulge).
     *
     * @tparam T
     * @param vertex
     * @return std::tuple<T, T, T>
     */
    template <typename T>
    [[nodiscard]] std::tuple<T, T, T> pvertex_to_tuple(const cavc::PlineVertex<T>& vertex)
    {
        return std::make_tuple(vertex.x(), vertex.y(), vertex.bulge());
    }
}

namespace fibomat
{
    /**
     * @brief Class represents a C0 curve build up by lines and circular arcs.
     *
     * Internally, the curve is saved as a carvc::Polyline, which itself save (x, y, bulge) tuples,
     * which represent the curve. See fibomat docs for a description of the bulge value.
     *
     * @tparam T underlying data type to be used.
     */
    template <typename T>
    class arc_spline
    {
    public:
        /**
         * @brief Construct a new empty curve object.
         *
         */
        arc_spline() = default;

        arc_spline(const arc_spline<T>& other) : m_curve(other.m_curve)
        {

        }

        /**
         * @brief Construct a new curve object from a cavc::Polyline.
         *
         * @param raw underlying polyline
         */
        arc_spline(const cavc::Polyline<T>& raw) : m_curve(raw)
        {
        }

        /**
         * @brief Construct a new curve object from a list of points (x, y, bulge)
         *
         * @param points    List of points
         * @param is_closed Indicator if the curve is closed. If true, the start end are assumed to be connected.
         */
        arc_spline(std::vector<std::tuple<T, T, T>>&& points, bool is_closed)
        {
            for (auto point : points)
            {
                m_curve.addVertex(std::get<0>(point), std::get<1>(point), std::get<2>(point));
            }

            m_curve.isClosed() = is_closed;
        }

        /**
         * @brief Construct a new curve object from a range.
         *
         * @tparam Rng
         * @param points    Range of points (x, y, bulge)
         * @param is_closed Indicator if the curve is closed. If true, the start end are assumed to be connected.
         */
        template <typename Rng>
        arc_spline(Rng&& points, bool is_closed)
        {
            for (auto point : points)
            {
                m_curve.addVertex(point[0], point[1], point[2]);
            }

            m_curve.isClosed() = is_closed;
        }

        /**
         * @brief Create a copy of the curve.
         *
         * @return curve
         */
        [[nodiscard]] arc_spline clone() const
        {
            return *this;
        }

        /**
         * @brief number of vertices in curve
         *
         * @return std::size_t
         */
        [[nodiscard]] std::size_t size() const
        {
            return m_curve.size();
        }

        /**
         * @brief Indicator if the curve is closed. If true, the start end are assumed to be connected.
         *
         * @return bool true if closed.
         */
        [[nodiscard]] bool is_closed() const
        {
            return m_curve.isClosed();
        }

        /**
         * @brief Bounding box of the curve.
         *
         * @return std::pair<std::pair<T, T>, std::pair<T, T>> bunding box of form ((xmin, ymin), (xmax, ymax))
         */
        [[nodiscard]] std::pair<std::pair<T, T>, std::pair<T, T>> bounding_box() const
        {
            auto bbox = cavc::getExtents(m_curve);

            // for (auto &v : m_curve.vertexes())
            // {
            //     std::cout << v.x() << " " << v.y() << " " << v.bulge() << ", ";
            // }

            // std::cout << "\n";

            // auto visitor = [&](std::size_t i, std::size_t j) {
            //     auto v = m_curve[i];
            //     std::cout << i << " " << j << " " << v.x() << " " << v.y() << " " << v.bulge() << ", ";
            //     return true;
            // };

            // m_curve.visitSegIndices(visitor);
            // std::cout << "\n";


            // std::cout << bbox.xMin << " " << bbox.yMin << " " << bbox.xMax  << " " << bbox.yMax << "\n";

            const auto& last_vertex = m_curve[m_curve.size() - 1];

            if (last_vertex.x() < bbox.xMin)
                bbox.xMin = last_vertex.x();
            if (last_vertex.y() < bbox.yMin)
                bbox.yMin = last_vertex.y();
            if (last_vertex.x() > bbox.xMax)
                bbox.xMax = last_vertex.x();
            if (last_vertex.y() > bbox.yMax)
                bbox.yMax = last_vertex.y();

            return{{bbox.xMin, bbox.yMin}, {bbox.xMax, bbox.yMax}};
        }

        /**
         * @brief Orientation of the curve
         *
         * @return bool true if orientation in mathematically positive direction
         */
        [[nodiscard]] bool orientation() const
        {
            if (!is_closed())
            {
                throw std::runtime_error("Cannot determine orientation if curve is not closed.");
            }

            if (m_curve.size() < 2)
            {
                throw std::runtime_error("Cannot determine orientation if curve has less than 2 points.");
            }

            return cavc::getArea(m_curve) > 0;
        }

        /**
         * @brief Length of curve
         *
         * @return T length
         */
        [[nodiscard]] T length() const
        {
            return cavc::getPathLength(m_curve);
        }

        [[nodiscard]] T area() const
        {
            return cavc::getArea(m_curve);
        }

        [[nodiscard]] std::pair<T, T> center() const
        {
            auto center = std::accumulate(
                std::begin(m_curve.vertexes()),
                std::end(m_curve.vertexes()),
                cavc::Vector2<T>::zero(),
                [](const auto& a, const auto& b) {return a + b.pos();}
            );

            center /= static_cast<T>(m_curve.size());

            return {center.x(), center.y()};
        }

        /**
         * @brief Start point of the curve
         *
         * @return std::tuple<T, T, T> start point point (x, y, bulge)
         */
        [[nodiscard]] std::tuple<T, T, T> start() const
        {
            if (m_curve.size() == 0)
            {
                throw std::runtime_error("An empty curve has no start point.");
            }

            return pvertex_to_tuple(m_curve.vertexes()[0]);
        }

        /**
         * @brief End point of the curve
         *
         * @return std::tuple<T, T, T> end point point (x, y, bulge)
         */
        [[nodiscard]] std::tuple<T, T, T> end() const
        {
            if (m_curve.size() == 0)
            {
                throw std::runtime_error("An empty curve has no end point.");
            }

            return pvertex_to_tuple(m_curve.lastVertex());
        }

        /**
         * @brief Curve vertices
         *
         * @return std::vector<std::tuple<T, T, T>> list of (x, y, bulge)
         */
        [[nodiscard]] std::vector<std::tuple<T, T, T>> vertices() const
        {
            std::vector<std::tuple<double, double, double>> vertices(m_curve.size());

            auto& raw_vertexes = m_curve.vertexes();

            for (std::size_t i = 0; i < m_curve.size(); ++i)
            {
                vertices[i] = pvertex_to_tuple(raw_vertexes[i]);
            }
            return vertices;
        }

        /**
         * @brief Add a point to curve. Only possible, if curve is not closed.
         *
         * @param x
         * @param y
         * @param bulge
         */
        void add_vertex(T x, T y, T bulge)
        {
            if (is_closed())
            {
                throw std::runtime_error("Adding vertex to closed curve.");
            }

            // TODO: check if curve is closed after adding?

            m_curve.addVertex(x, y, bulge);
        }

        /**
         * @brief Add a point to curve. Only possible, if curve is not closed.
         *
         * @param cavc::Pvertex
         */
        void add_vertex(cavc::PlineVertex<T> vertex)
        {
            if (is_closed())
            {
                throw std::runtime_error("Adding vertex to closed curve.");
            }

            // TODO: check if curve is closed after adding?

            m_curve.addVertex(vertex);
        }

        /**
         * @brief Add a range of vertices to the curve
         *
         * @tparam Rng
         * @param points Range of points (x, y, bulge)
         */
        template <typename Rng>
        void add_vertices(Rng&& points)
        {
            if (is_closed())
            {
                throw std::runtime_error("Adding vertex to closed curve.");
            }

            for (auto point : points)
            {
                m_curve.addVertex(point[0], point[1], point[2]);
            }
        }

        /**
         * @brief Reverse th curve in-place.
         *
         */
        void reverse()
        {
            cavc::invertDirection(m_curve);
        }

        /**
         * @brief Translate the curve by vec.
         *
         * @param vec
         */
        void translate(const cavc::Vector2<T>& vec)
        {
            cavc::translatePolyline(m_curve, vec);
        }

        /**
         * @brief Scale the curve by s.
         *
         * @param s
         */
        void scale(T s)
        {
            cavc::scalePolyline(m_curve, s);
        }

        /**
         * @brief Rotate the curve about `angle` around the origin.
         *
         * @param angle If positive, rotation is in math. positive direction.
         */
        void rotate(T angle)
        {
            T cos = std::cos(angle);
            T sin = std::sin(angle);

            for (auto &v : m_curve.vertexes())
            {
                v = cavc::PlineVertex<double>(cos * v.x() - sin * v.y(), sin * v.x() + cos * v.y(), v.bulge());
            }
        }

        void mirror(const cavc::Vector2<T>& axis)
        {
            T norm = cavc::length(axis);
            T norm2 = norm * norm;

            if (cavc::utils::fuzzyEqual(norm, 0.))
            {
                throw std::runtime_error("mirror axis may not be the null vector");
            }

            T a = (axis.x()*axis.x() - axis.y()*axis.y()) / norm2;
            T b = (2*axis.x()*axis.y()) / norm2;
            T c = b;
            T d = (axis.y()*axis.y() - axis.x()*axis.x()) / norm2;

            for (auto &v : m_curve.vertexes())
            {
                v = cavc::PlineVertex<double>(a * v.x() + b * v.y(), c * v.x() + d * v.y(), -v.bulge());
            }

        }

        /**
         * @brief Visit all segments of the curve.
         *
         * @tparam Visitor
         * @param visitor Function with signature: `(int segment_index, tuple seg_start_point, tuple seg_end_point) -> bool`.
         *                If func. returns false, iteration is stopped immediately.
         */
        template <typename Visitor>
        void visit(Visitor&& visitor) const
        {
            auto wrapped_visitor = [&] (std::size_t i, std::size_t j)
            {
                static std::size_t i_seg = 0;

                return visitor(i_seg++, pvertex_to_tuple(m_curve.vertexes()[i]), pvertex_to_tuple(m_curve.vertexes()[j]));
            };

            m_curve.visitSegIndices(wrapped_visitor);
        }

        /**
         * @brief Check if point lies in the enclosed are of a curve. Curve must be closed.
         *
         * @param x
         * @param y
         * @return bool
         */
        [[nodiscard]] bool contains(T x, T y) const
        {
            if (!is_closed())
            {
                throw std::runtime_error("Curve is not closed, hence it cannot be checked if it contains something.");
            }

            return getWindingNumber(m_curve, cavc::Vector2<double>(x, y)) != 0;
        }

        [[nodiscard]] std::tuple<std::size_t, std::tuple<T, T>, T> closest_point(T x, T y) const
        /**
         * @brief returns the closest point of (x, y) on the arc
         * @param x
         * @param y
         * @return tuple: segment index, point, distance((x, y), point)
         */
        {
            cavc::ClosestPoint<T> closestPoint(m_curve, cavc::Vector2<T>(x, y));

            const auto& point = closestPoint.point();

            return {closestPoint.index(), {point.x(), point.y()}, closestPoint.distance()};
        }

        /**
         * @brief Return underlying cavc::Polyline.
         *
         * @return cavc::Polyline<T>&
         */
        [[nodiscard]] cavc::Polyline<T>& raw_curve()
        {
            return m_curve;
        }

        /**
         * @brief Return underlying cavc::Polyline.
         *
         * @return cavc::Polyline<T>&
         */
        [[nodiscard]] const cavc::Polyline<T>& raw_curve() const
        {
            return m_curve;
        }

    private:
        cavc::Polyline<T> m_curve;
    };
}

#endif //FIB_O_MAT_CURVE_H
