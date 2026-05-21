// Canny edge detector built from a 16-tap Gaussian and its derivative.
// Selects the high threshold from a histogram of the normalised
// magnitude; weak edges are every pixel that survives non-maximum
// suppression. Hysteresis runs through hysteresis_flood_fill.
//
// From Fuhl, W., Kübler, T., Sippel, K., Rosenstiel, W., Kasneci, E.
// (2015). "ExCuSe: Robust Pupil Detection in Real-World Scenarios."
// *CAIP 2015*, 39-51.

#pragma once

#include "cheshm/edges/edge_hysteresis.hpp"

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace cheshm
{

// Returns the binary edge map. ``image`` is grayscale; any depth is
// accepted and converted to float internally without mutating the
// caller's matrix. ``non_edge_pixel_count`` is the histogram-cumulative
// cutoff that fixes the high threshold: the first bin whose running
// count exceeds this value becomes ``high_th = (i + 1) / bins``.
// ``magnitude_out``, when non-null, receives the normalised float
// magnitude image.
inline cv::Mat
canny_gaussian16(const cv::Mat& image, int non_edge_pixel_count, int bins, cv::Mat* magnitude_out = nullptr)
{
    constexpr int k_sz = 16;

    static const float gau[k_sz] = {
        0.000000220358050f,
        0.000007297256405f,
        0.000146569312970f,
        0.001785579770079f,
        0.013193749090229f,
        0.059130281094460f,
        0.160732768610747f,
        0.265003534507060f,
        0.265003534507060f,
        0.160732768610747f,
        0.059130281094460f,
        0.013193749090229f,
        0.001785579770079f,
        0.000146569312970f,
        0.000007297256405f,
        0.000000220358050f,
    };
    static const float deriv_gau[k_sz] = {
        -0.000026704586264f,
        -0.000276122963398f,
        -0.003355163265098f,
        -0.024616683775044f,
        -0.108194751875585f,
        -0.278368310241814f,
        -0.388430056419619f,
        -0.196732206873178f,
        0.196732206873178f,
        0.388430056419619f,
        0.278368310241814f,
        0.108194751875585f,
        0.024616683775044f,
        0.003355163265098f,
        0.000276122963398f,
        0.000026704586264f,
    };

    cv::Mat work;
    image.convertTo(work, CV_32FC1);

    const cv::Mat gau_row(1, k_sz, CV_32FC1, const_cast<float*>(gau));
    const cv::Mat deriv_row(1, k_sz, CV_32FC1, const_cast<float*>(deriv_gau));

    cv::Mat res_x;
    cv::Mat res_y;
    cv::sepFilter2D(work, res_x, CV_32F, deriv_row, gau_row.t(), cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::sepFilter2D(work, res_y, CV_32F, gau_row, deriv_row.t(), cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    cv::Mat magnitude;
    cv::magnitude(res_x, res_y, magnitude);

    cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX, CV_32FC1);
    cv::Mat res_idx;
    cv::normalize(magnitude, res_idx, 0, bins - 1, cv::NORM_MINMAX, CV_32S);

    cv::Mat res_idx_f32;
    res_idx.convertTo(res_idx_f32, CV_32F);
    const std::vector<int> channels = {0};
    const std::vector<int> hist_size = {bins};
    const std::vector<float> ranges = {0.0f, static_cast<float>(bins)};
    cv::Mat hist_mat;
    cv::calcHist(std::vector<cv::Mat>{res_idx_f32}, channels, cv::Mat(), hist_mat, hist_size, ranges);

    float high_th = 0.0f;
    int sum = 0;
    for (int i = 0; i < bins; ++i)
    {
        sum += static_cast<int>(hist_mat.at<float>(i));
        if (sum > non_edge_pixel_count)
        {
            high_th = static_cast<float>(i + 1) / static_cast<float>(bins);
            break;
        }
    }

    cv::Mat non_ms = cv::Mat::zeros(work.rows, work.cols, CV_8U);
    cv::Mat non_ms_hth = cv::Mat::zeros(work.rows, work.cols, CV_8U);

    for (int i = 1; i < magnitude.rows - 1; ++i)
    {
        char* p_non_ms = non_ms.ptr<char>(i);
        char* p_non_ms_hth = non_ms_hth.ptr<char>(i);

        const float* p_res = magnitude.ptr<float>(i);
        const float* p_res_t = magnitude.ptr<float>(i - 1);
        const float* p_res_b = magnitude.ptr<float>(i + 1);

        const float* p_x = res_x.ptr<float>(i);
        const float* p_y = res_y.ptr<float>(i);

        for (int j = 1; j < magnitude.cols - 1; ++j)
        {
            const float iy = p_y[j];
            const float ix = p_x[j];
            float grad1 = 0.0f;
            float grad2 = 0.0f;
            float d = 0.0f;

            if ((iy <= 0 && ix > -iy) || (iy >= 0 && ix < -iy))
            {
                d = std::abs(iy / ix);
                grad1 = (p_res[j + 1] * (1 - d)) + (p_res_t[j + 1] * d);
                grad2 = (p_res[j - 1] * (1 - d)) + (p_res_b[j - 1] * d);
                if (p_res[j] >= grad1 && p_res[j] >= grad2)
                {
                    p_non_ms[j] = static_cast<char>(255);
                    if (p_res[j] > high_th)
                        p_non_ms_hth[j] = static_cast<char>(255);
                }
            }

            if ((ix > 0 && -iy >= ix) || (ix < 0 && -iy <= ix))
            {
                d = std::abs(ix / iy);
                grad1 = (p_res_t[j] * (1 - d)) + (p_res_t[j + 1] * d);
                grad2 = (p_res_b[j] * (1 - d)) + (p_res_b[j - 1] * d);
                if (p_res[j] >= grad1 && p_res[j] >= grad2)
                {
                    p_non_ms[j] = static_cast<char>(255);
                    if (p_res[j] > high_th)
                        p_non_ms_hth[j] = static_cast<char>(255);
                }
            }

            if ((ix <= 0 && ix > iy) || (ix >= 0 && ix < iy))
            {
                d = std::abs(ix / iy);
                grad1 = (p_res_t[j] * (1 - d)) + (p_res_t[j - 1] * d);
                grad2 = (p_res_b[j] * (1 - d)) + (p_res_b[j + 1] * d);
                if (p_res[j] >= grad1 && p_res[j] >= grad2)
                {
                    p_non_ms[j] = static_cast<char>(255);
                    if (p_res[j] > high_th)
                        p_non_ms_hth[j] = static_cast<char>(255);
                }
            }

            if ((iy < 0 && ix <= iy) || (iy > 0 && ix >= iy))
            {
                d = std::abs(iy / ix);
                grad1 = (p_res[j - 1] * (1 - d)) + (p_res_t[j - 1] * d);
                grad2 = (p_res[j + 1] * (1 - d)) + (p_res_b[j + 1] * d);
                if (p_res[j] >= grad1 && p_res[j] >= grad2)
                {
                    p_non_ms[j] = static_cast<char>(255);
                    if (p_res[j] > high_th)
                        p_non_ms_hth[j] = static_cast<char>(255);
                }
            }
        }
    }

    if (magnitude_out)
        *magnitude_out = magnitude;

    return hysteresis_flood_fill(non_ms_hth, non_ms, 10000);
}

} // namespace cheshm
