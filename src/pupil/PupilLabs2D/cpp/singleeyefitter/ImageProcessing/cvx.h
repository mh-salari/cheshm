#ifndef __SINGLEEYEFITTER_CVX_H__
#define __SINGLEEYEFITTER_CVX_H__

#include <opencv2/core.hpp>

namespace singleeyefitter::cvx
{

// `cv::findNonZero` crashes when the input contains no non-zero pixels
// in older OpenCV builds. Temporarily writes a 1 at (0, 0) before the
// call and restores it to 0 afterwards.
inline void findNonZero(cv::Mat& in, cv::OutputArray& out)
{
    in.at<uchar>(0, 0) = 1;
    cv::findNonZero(in, out);
    in.at<uchar>(0, 0) = 0;
}

} // namespace singleeyefitter::cvx

#endif // __SINGLEEYEFITTER_CVX_H__
