#ifndef __UTILS_H__
#define __UTILS_H__

#include "common/constants.h"
#include "geometry/Ellipse.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace singleeyefitter
{

template <typename Scalar>
inline Eigen::Matrix<Scalar, 2, 1> toEigen(const cv::Point2f& point)
{
    return Eigen::Matrix<Scalar, 2, 1>(static_cast<Scalar>(point.x), static_cast<Scalar>(point.y));
}

template <typename Scalar>
inline Ellipse2D<Scalar> toEllipse(const cv::RotatedRect& rect)
{
    return Ellipse2D<Scalar>(toEigen<Scalar>(rect.center),
                             static_cast<Scalar>(rect.size.height / 2.0),
                             static_cast<Scalar>(rect.size.width / 2.0),
                             static_cast<Scalar>((rect.angle + 90.0) * constants::PI / 180.0));
}

} // namespace singleeyefitter

#endif // __UTILS_H__
