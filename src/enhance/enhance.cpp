#include "cheshm/enhance/enhance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <opencv2/imgproc.hpp>

namespace cheshm::enhance
{

cv::Mat clahe(const cv::Mat& gray, double clip_limit, int tile)
{
    const int t = std::max(1, tile);
    cv::Ptr<cv::CLAHE> op = cv::createCLAHE(clip_limit, cv::Size(t, t));
    cv::Mat out;
    op->apply(gray, out);
    return out;
}

cv::Mat percentile_stretch(const cv::Mat& gray, double lo_pct, double hi_pct)
{
    std::array<long, 256> hist{};
    for (int y = 0; y < gray.rows; ++y)
    {
        const std::uint8_t* row = gray.ptr<std::uint8_t>(y);
        for (int x = 0; x < gray.cols; ++x)
            ++hist[row[x]];
    }

    const double total = static_cast<double>(gray.rows) * gray.cols;
    const double lo_count = std::clamp(lo_pct, 0.0, 100.0) / 100.0 * total;
    const double hi_count = std::clamp(hi_pct, 0.0, 100.0) / 100.0 * total;

    int lo = 0;
    int hi = 255;
    double cum = 0.0;
    for (int i = 0; i < 256; ++i)
    {
        cum += static_cast<double>(hist[i]);
        if (cum >= lo_count)
        {
            lo = i;
            break;
        }
    }
    cum = 0.0;
    for (int i = 0; i < 256; ++i)
    {
        cum += static_cast<double>(hist[i]);
        if (cum >= hi_count)
        {
            hi = i;
            break;
        }
    }

    if (hi <= lo)
        return gray.clone();

    const double scale = 255.0 / static_cast<double>(hi - lo);
    cv::Mat out;
    gray.convertTo(out, CV_8U, scale, -lo * scale);
    return out;
}

cv::Mat gamma(const cv::Mat& gray, double g)
{
    const double exponent = (g > 0.0) ? g : 1.0;
    cv::Mat lut(1, 256, CV_8U);
    std::uint8_t* p = lut.ptr<std::uint8_t>();
    for (int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<std::uint8_t>(std::pow(i / 255.0, exponent) * 255.0);

    cv::Mat out;
    cv::LUT(gray, lut, out);
    return out;
}

cv::Mat bilateral(const cv::Mat& gray, int d, double sigma_color, double sigma_space)
{
    cv::Mat out;
    cv::bilateralFilter(gray, out, d, sigma_color, sigma_space);
    return out;
}

cv::Mat unsharp(const cv::Mat& gray, double sigma, double amount)
{
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(0, 0), sigma);
    cv::Mat out;
    cv::addWeighted(gray, 1.0 + amount, blurred, -amount, 0.0, out);
    return out;
}

} // namespace cheshm::enhance
