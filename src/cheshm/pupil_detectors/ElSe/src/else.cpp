// ElSe pupil detector algorithm body.

#include "ElSe/else.hpp"

#include "cheshm/canny.hpp"
#include "cheshm/canny_gaussian16.hpp"
#include "cheshm/ellipse_intensity_gap.hpp"

#include "ElSe/defaults.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>

namespace cheshm::ElSe
{
namespace
{

using namespace cv;  // NOLINT(google-build-using-namespace)
using namespace std; // NOLINT(google-build-using-namespace)

using defaults::IMG_SIZE;

// Intensity gap (inside-ellipse mean vs surrounding-band mean) below
// which a candidate ellipse is rejected by the ellipse-quality gates.
constexpr float INNER_OUTER_INTENSITY_GAP_MIN = 10.0f;

// Returns true when the ellipse looks pupil-like: its interior is at
// least INNER_OUTER_INTENSITY_GAP_MIN darker than the surrounding band.
// erg receives the integer-rounded mean difference.
bool is_good_ellipse_eval(const RotatedRect& ellipse, const Mat& pic, int& erg)
{
    if (ellipse.center.x == 0 && ellipse.center.y == 0)
        return false;

    const float x0 = ellipse.center.x;
    const float y0 = ellipse.center.y;

    const cheshm::PixelBox inner = {
        static_cast<int>(std::ceil(x0 - ellipse.size.width / 4.0)),
        static_cast<int>(std::floor(x0 + ellipse.size.width / 4.0)),
        static_cast<int>(std::ceil(y0 - ellipse.size.height / 4.0)),
        static_cast<int>(std::floor(y0 + ellipse.size.height / 4.0)),
    };
    const cheshm::PixelBox outer = {
        static_cast<int>(x0 - ellipse.size.width * 0.75),
        static_cast<int>(x0 + ellipse.size.width * 0.75),
        static_cast<int>(y0 - ellipse.size.height * 0.75),
        static_cast<int>(y0 + ellipse.size.height * 0.75),
    };
    const cheshm::PixelBox cutout = {
        static_cast<int>(std::ceil(x0 - ellipse.size.width / 2)),
        static_cast<int>(std::floor(x0 + ellipse.size.width / 2)) + 1,
        static_cast<int>(std::ceil(y0 - ellipse.size.height / 2)),
        static_cast<int>(std::floor(y0 + ellipse.size.height / 2)) + 1,
    };

    const auto r = cheshm::check_ellipse_intensity_gap(pic, inner, outer, cutout, INNER_OUTER_INTENSITY_GAP_MIN);
    erg = static_cast<int>(r.diff);
    return r.passes;
}

// Returns the mean intensity sampled at scaled-down copies of the
// curve (95% down to 80% of each point's vector to the ellipse centre,
// in 1% steps). Lower means a darker interior, which scores as a
// stronger pupil candidate in get_curves.
int calc_inner_gray(const Mat& pic, const std::vector<Point>& curve, const RotatedRect& ellipse)
{
    int gray_val = 0;
    int gray_cnt = 0;

    Mat checkmap = Mat::zeros(pic.size(), CV_8U);

    for (const Point& point : curve)
    {
        int vec_x = static_cast<int>(round(point.x - ellipse.center.x));
        int vec_y = static_cast<int>(round(point.y - ellipse.center.y));

        for (float p = 0.95f; p > 0.80f; p -= 0.01f)
        {
            int p_x = static_cast<int>(round(ellipse.center.x + float((float(vec_x) * p) + 0.5)));
            int p_y = static_cast<int>(round(ellipse.center.y + float((float(vec_y) * p) + 0.5)));

            if (p_x > 0 && p_x < pic.cols && p_y > 0 && p_y < pic.rows)
            {
                if (checkmap.data[(pic.cols * p_y) + p_x] == 0)
                {
                    checkmap.data[(pic.cols * p_y) + p_x] = 1;
                    gray_val += static_cast<unsigned int>(pic.data[(pic.cols * p_y) + p_x]);
                    gray_cnt++;
                }
            }
        }
    }

    if (gray_cnt > 0)
        gray_val = gray_val / gray_cnt;
    else
        gray_val = 1000;

    return gray_val;
}

// Walks the edge image, collects connected curves, fits an ellipse to
// each, gates by shape and pupil-area bounds, and keeps the single
// best curve under the inner-gray score (or no curve if none pass).
std::vector<std::vector<Point>> get_curves(const Mat& pic,
                                           const Mat& edge,
                                           int start_x,
                                           int end_x,
                                           int start_y,
                                           int end_y,
                                           double mean_dist,
                                           int inner_color_range,
                                           float min_area,
                                           float max_area)
{
    if (start_x < 2)
        start_x = 2;
    if (start_y < 2)
        start_y = 2;
    if (end_x > pic.cols - 2)
        end_x = pic.cols - 2;
    if (end_y > pic.rows - 2)
        end_y = pic.rows - 2;

    const cv::Rect walk_roi(start_x, start_y, end_x - start_x, end_y - start_y);
    cv::Mat labels;
    const int n_labels = cv::connectedComponents(edge(walk_roi), labels, 8, CV_32S);

    // Bucket every foreground pixel under its component label, in row-major
    // scan order. Component 0 is the background and is skipped.
    std::vector<std::vector<Point>> components(n_labels > 0 ? n_labels : 0);
    for (int j = 0; j < labels.rows; ++j)
    {
        const int32_t* row = labels.ptr<int32_t>(j);
        for (int i = 0; i < labels.cols; ++i)
        {
            const int label = row[i];
            if (label != 0)
                components[label].emplace_back(i + start_x, j + start_y);
        }
    }

    std::vector<std::vector<Point>> all_lines;
    std::vector<Point> all_means;
    all_lines.reserve(n_labels);
    all_means.reserve(n_labels);
    for (int label = 1; label < n_labels; ++label)
    {
        auto& curve = components[label];
        if (curve.size() <= 10)
            continue;
        long long sx = 0, sy = 0;
        for (const auto& p : curve)
        {
            sx += p.x;
            sy += p.y;
        }
        const auto n = static_cast<double>(curve.size());
        all_means.emplace_back(static_cast<int>(std::floor(sx / n + 0.5)), static_cast<int>(std::floor(sy / n + 0.5)));
        all_lines.push_back(std::move(curve));
    }

    std::vector<std::vector<Point>> all_curves;
    Point mean_p;
    bool add_curve;
    int mean_inner_gray;
    int mean_inner_gray_last = 1000000;
    RotatedRect selected_ellipse;

    for (std::size_t iii = 0; iii < all_lines.size(); ++iii)
    {
        std::vector<Point>& curve = all_lines[iii];
        mean_p = all_means[iii];

        int results = 0;
        add_curve = true;

        RotatedRect ellipse;

        for (const Point& point : curve)
            if (abs(mean_p.x - point.x) <= mean_dist && abs(mean_p.y - point.y) <= mean_dist)
                add_curve = false;

        // is ellipse fit possible
        if (add_curve)
        {
            ellipse = fitEllipse(Mat(curve));

            if (ellipse.center.x < 0 || ellipse.center.y < 0 || ellipse.center.x > pic.cols ||
                ellipse.center.y > pic.rows)
            {
                add_curve = false;
            }

            if (ellipse.size.height > 3 * ellipse.size.width || ellipse.size.width > 3 * ellipse.size.height)
            {
                add_curve = false;
            }

            if (add_curve)
            { // pupil area
                if (ellipse.size.width * ellipse.size.height < min_area ||
                    ellipse.size.width * ellipse.size.height > max_area)
                    add_curve = false;
            }

            if (add_curve)
            {
                if (!is_good_ellipse_eval(ellipse, pic, results))
                    add_curve = false;
            }
        }

        if (add_curve)
        {
            if (inner_color_range >= 0)
            {
                mean_inner_gray = 0;
                mean_inner_gray = calc_inner_gray(pic, curve, ellipse);
                mean_inner_gray =
                    static_cast<int>(mean_inner_gray * (1 + abs(ellipse.size.height - ellipse.size.width)));

                if (mean_inner_gray_last > mean_inner_gray)
                {
                    mean_inner_gray_last = mean_inner_gray;
                    all_curves.clear();
                    all_curves.push_back(curve);
                }
                else if (mean_inner_gray_last == mean_inner_gray)
                {
                    if (curve.size() > all_curves[0].size())
                    {
                        mean_inner_gray_last = mean_inner_gray;
                        all_curves.clear();
                        all_curves.push_back(curve);
                        selected_ellipse = ellipse;
                    }
                }
            }
        }
    }

    return all_curves;
}

// Picks the surviving curve from get_curves and refits an ellipse to
// it. Returns a zero-RotatedRect when get_curves yields zero or
// multiple curves.
RotatedRect find_best_edge(const Mat& pic,
                           const Mat& edge,
                           int start_x,
                           int end_x,
                           int start_y,
                           int end_y,
                           double mean_dist,
                           int inner_color_range,
                           float min_area,
                           float max_area)
{
    const auto all_curves =
        get_curves(pic, edge, start_x, end_x, start_y, end_y, mean_dist, inner_color_range, min_area, max_area);
    if (all_curves.size() != 1)
        return RotatedRect{};

    RotatedRect ellipse = fitEllipse(Mat(all_curves[0]));
    if (ellipse.center.x < 0 || ellipse.center.y < 0 || ellipse.center.x > pic.cols ||
        ellipse.center.y > pic.rows)
        return RotatedRect{};
    return ellipse;
}


// Coarse downsample + dark-pixel mean. For each output cell averages
// the input window centred on the corresponding source pixel, then
// re-averages only the pixels at or below that local mean. Used by
// blob_finder to suppress glints before the blob template runs.
void mum(const Mat& pic, Mat& result, int fak)
{
    int fak_ges = fak + 1;
    int sz_x = pic.cols / fak_ges;
    int sz_y = pic.rows / fak_ges;

    result = Mat::zeros(sz_y, sz_x, CV_8U);

    int hist[256];
    int mean = 0;
    int cnt = 0;
    int mean_2 = 0;

    int idx = 0;
    int idy = 0;

    for (int i = 0; i < sz_y; i++)
    {
        idy += fak_ges;

        for (int j = 0; j < sz_x; j++)
        {
            idx += fak_ges;

            for (int k = 0; k < 256; k++)
                hist[k] = 0;

            mean = 0;
            cnt = 0;

            for (int ii = -fak; ii <= fak; ii++)
                for (int jj = -fak; jj <= fak; jj++)
                {
                    if (idy + ii > 0 && idy + ii < pic.rows && idx + jj > 0 && idx + jj < pic.cols)
                    {
                        const auto v = pic.data[(pic.cols * (idy + ii)) + (idx + jj)];
                        hist[v]++;
                        cnt++;
                        mean += v;
                    }
                }
            mean = mean / cnt;

            mean_2 = 0;
            cnt = 0;
            for (int ii = 0; ii <= mean; ii++)
            {
                mean_2 += ii * hist[ii];
                cnt += hist[ii];
            }

            if (cnt == 0)
                mean_2 = mean;
            else
                mean_2 = mean_2 / cnt;

            result.data[(sz_x * (i)) + (j)] = mean_2;
        }
        idx = 0;
    }
}

// Builds a (4*rad+1)^2 disk-shaped convolution template. Negative
// weights form the disk of radius `rad` at the centre, positive
// weights outside. `all_mat_neg` carries the negative band only,
// used for the dual-filter scoring inside blob_finder.
void gen_blob_neu(int rad, Mat& all_mat, Mat& all_mat_neg)
{
    int len = 1 + (4 * rad);
    int c0 = rad * 2;
    float negis = 0;
    float posis = 0;

    all_mat = Mat::zeros(len, len, CV_32FC1);
    all_mat_neg = Mat::zeros(len, len, CV_32FC1);

    float *p, *p_neg;
    for (int i = -rad * 2; i <= rad * 2; i++)
    {
        p = all_mat.ptr<float>(c0 + i);

        for (int j = -rad * 2; j <= rad * 2; j++)
        {
            if (i < -rad || i > rad)
            { // positive band
                p[c0 + j] = 1;
                posis++;
            }
            else
            { // disk interior

                int sz_w = static_cast<int>(sqrt(float(rad * rad) - float(i * i)));

                if (abs(j) <= sz_w)
                {
                    p[c0 + j] = -1;
                    negis++;
                }
                else
                {
                    p[c0 + j] = 1;
                    posis++;
                }
            }
        }
    }

    for (int i = 0; i < len; i++)
    {
        p = all_mat.ptr<float>(i);
        p_neg = all_mat_neg.ptr<float>(i);

        for (int j = 0; j < len; j++)
        {
            if (p[j] > 0)
            {
                p[j] = 1.0f / posis;
                p_neg[j] = 0.0f;
            }
            else
            {
                p[j] = -1.0f / negis;
                p_neg[j] = 1.0f / negis;
            }
        }
    }
}

// Stricter variant of is_good_ellipse_eval used inside blob_finder to
// validate the coarse blob position before accepting it. Same
// inside-vs-outside intensity-gap test, no `erg` output param.
bool is_good_ellipse_evaluation(const RotatedRect& ellipse, const Mat& pic)
{
    if (ellipse.center.x == 0 && ellipse.center.y == 0)
        return false;

    const float x0 = ellipse.center.x;
    const float y0 = ellipse.center.y;

    const cheshm::PixelBox inner = {
        static_cast<int>(std::ceil(x0 - ellipse.size.width / 4.0)),
        static_cast<int>(std::floor(x0 + ellipse.size.width / 4.0)),
        static_cast<int>(std::ceil(y0 - ellipse.size.height / 4.0)),
        static_cast<int>(std::floor(y0 + ellipse.size.height / 4.0)),
    };
    const cheshm::PixelBox outer = {
        static_cast<int>(std::ceil(x0 - ellipse.size.width * 0.75)),
        static_cast<int>(std::floor(x0 + ellipse.size.width * 0.75)),
        static_cast<int>(std::ceil(y0 - ellipse.size.height * 0.75)),
        static_cast<int>(std::floor(y0 + ellipse.size.height * 0.75)),
    };
    const cheshm::PixelBox cutout = {
        static_cast<int>(std::ceil(x0 - ellipse.size.width / 2)),
        static_cast<int>(std::floor(x0 + ellipse.size.width / 2)) + 1,
        static_cast<int>(std::ceil(y0 - ellipse.size.height / 2)),
        static_cast<int>(std::floor(y0 + ellipse.size.height / 2)) + 1,
    };

    return cheshm::check_ellipse_intensity_gap(pic, inner, outer, cutout, INNER_OUTER_INTENSITY_GAP_MIN).passes;
}

// Coarse-blob fallback. Convolves a circular template against the
// mum-reduced image, picks the maximum response position, refines to a
// dark-pixel centroid, and returns it as a degenerate RotatedRect
// (size = template). Used when the edge-based path produces no valid
// ellipse.
RotatedRect blob_finder(const Mat& pic)
{
    Point pos(0, 0);

    const int fak_mum = 5;
    const int fakk = pic.cols > pic.rows ? (pic.cols / 100) + 1 : (pic.rows / 100) + 1;

    Mat img;
    mum(pic, img, fak_mum);
    img.convertTo(img, CV_32FC1);

    Mat blob_mat, blob_mat_neg;
    gen_blob_neu(fakk, blob_mat, blob_mat_neg);

    Mat result, result_neg, erg;
    filter2D(img, result, -1, blob_mat, Point(-1, -1), 0, BORDER_REPLICATE);
    cv::max(result, 0.0, result);

    filter2D(img, result_neg, -1, blob_mat_neg, Point(-1, -1), 0, BORDER_REPLICATE);
    cv::subtract(cv::Scalar(255.0f), result_neg, result_neg);
    cv::multiply(result_neg, result, erg);

    double max_val = 0.0;
    cv::Point max_loc;
    cv::minMaxLoc(erg, nullptr, &max_val, nullptr, &max_loc);
    if (max_val > 0.0)
    {
        pos.x = (fak_mum + 1) + (max_loc.x * (fak_mum + 1));
        pos.y = (fak_mum + 1) + (max_loc.y * (fak_mum + 1));
    }

    if (pos.y > 0 && pos.y < pic.rows && pos.x > 0 && pos.x < pic.cols)
    {
        // dark-pixel centroid refinement around the response peak.
        int opti_x = 0;
        int opti_y = 0;

        float mm = 0;
        float cnt = 0;
        for (int i = -(2); i < (2); i++)
        {
            for (int j = -(2); j < (2); j++)
            {
                if (pos.y + i > 0 && pos.y + i < pic.rows && pos.x + j > 0 && pos.x + j < pic.cols)
                {
                    mm += pic.data[(pic.cols * (pos.y + i)) + (pos.x + j)];
                    cnt++;
                }
            }
        }

        if (cnt > 0)
            mm = ceil(mm / cnt);

        int th_bot = 0;
        if (pos.y > 0 && pos.y < pic.rows && pos.x > 0 && pos.x < pic.cols)
            th_bot = static_cast<int>(pic.data[(pic.cols * (pos.y)) + (pos.x)] +
                                      abs(mm - pic.data[(pic.cols * (pos.y)) + (pos.x)]));
        cnt = 0;

        for (int i = -(fak_mum * fak_mum); i < (fak_mum * fak_mum); i++)
        {
            for (int j = -(fak_mum * fak_mum); j < (fak_mum * fak_mum); j++)
            {
                if (pos.y + i > 0 && pos.y + i < pic.rows && pos.x + j > 0 && pos.x + j < pic.cols)
                {
                    if (pic.data[(pic.cols * (pos.y + i)) + (pos.x + j)] <= th_bot)
                    {
                        opti_x += pos.x + j;
                        opti_y += pos.y + i;
                        cnt++;
                    }
                }
            }
        }

        if (cnt > 0)
        {
            opti_x = static_cast<int>(opti_x / cnt);
            opti_y = static_cast<int>(opti_y / cnt);
        }
        else
        {
            opti_x = pos.x;
            opti_y = pos.y;
        }

        pos.x = opti_x;
        pos.y = opti_y;
    }

    RotatedRect ellipse;

    if (pos.y > 0 && pos.y < pic.rows && pos.x > 0 && pos.x < pic.cols)
    {
        ellipse.center.x = static_cast<float>(pos.x);
        ellipse.center.y = static_cast<float>(pos.y);
        ellipse.angle = 0.0;
        ellipse.size.height = static_cast<float>((fak_mum * fak_mum * 2) + 1);
        ellipse.size.width = static_cast<float>((fak_mum * fak_mum * 2) + 1);

        if (!is_good_ellipse_evaluation(ellipse, pic))
        {
            ellipse.center.x = 0;
            ellipse.center.y = 0;
            ellipse.angle = 0;
            ellipse.size.height = 0;
            ellipse.size.width = 0;
        }
    }
    else
    {
        ellipse.center.x = 0;
        ellipse.center.y = 0;
        ellipse.angle = 0;
        ellipse.size.height = 0;
        ellipse.size.width = 0;
    }

    return ellipse;
}

} // namespace

std::optional<DetectResult> detect(const cv::Mat& frame, float min_area_ratio, float max_area_ratio)
{
    cv::Mat downscaled = frame;
    float scaling_ratio = 1.0f;
    if (frame.rows > IMG_SIZE || frame.cols > IMG_SIZE)
    {
        float rw = static_cast<float>(IMG_SIZE) / static_cast<float>(frame.cols);
        float rh = static_cast<float>(IMG_SIZE) / static_cast<float>(frame.rows);
        scaling_ratio = std::min<float>(std::min<float>(rw, rh), 1.0f);
        cv::resize(frame, downscaled, cv::Size(), scaling_ratio, scaling_ratio, cv::INTER_LINEAR);
    }

    cv::Mat pic;
    cv::normalize(downscaled, pic, 0, 255, cv::NORM_MINMAX, CV_8U);

    const float min_area = downscaled.cols * downscaled.rows * min_area_ratio;
    const float max_area = downscaled.cols * downscaled.rows * max_area_ratio;

    constexpr double border = 0.0; // ROI handling is delegated to the binding layer.
    constexpr double mean_dist = 3;
    constexpr int inner_color_range = 0;

    const int start_x = static_cast<int>(std::floor(static_cast<double>(pic.cols) * border));
    const int start_y = static_cast<int>(std::floor(static_cast<double>(pic.rows) * border));

    const int end_x = pic.cols - start_x;
    const int end_y = pic.rows - start_y;

    const cv::Rect inner_rect(start_x, start_y, end_x - start_x, end_y - start_y);
    const cv::Mat picpic = pic(inner_rect);

    const int non_edge_pixel_count = static_cast<int>(std::round(0.7 * picpic.cols * picpic.rows));
    const cv::Mat detected_edges2 = cheshm::canny_gaussian16(picpic, non_edge_pixel_count, 64);

    cv::Mat detected_edges = cv::Mat::zeros(pic.rows, pic.cols, CV_8U);
    detected_edges2.copyTo(detected_edges(inner_rect));

    cheshm::filter_edges(detected_edges, start_x, end_x, start_y, end_y);

    cv::RotatedRect ellipse = find_best_edge(pic,
                                             detected_edges,
                                             start_x,
                                             end_x,
                                             start_y,
                                             end_y,
                                             mean_dist,
                                             inner_color_range,
                                             min_area,
                                             max_area);

    DetectionMethod method;
    const bool primary_invalid = (ellipse.center.x <= 0 && ellipse.center.y <= 0) || ellipse.center.x >= pic.cols ||
                                 ellipse.center.y >= pic.rows;

    if (primary_invalid)
    {
        method = DetectionMethod::BlobFallback;
        ellipse = blob_finder(pic);
        ellipse.angle = 0;
        ellipse.size = cv::Size(0, 0);

        if (ellipse.center.x <= 0 && ellipse.center.y <= 0)
        {
            // Both paths failed; surface as "no detection".
            return std::nullopt;
        }
    }
    else
    {
        method = DetectionMethod::Ellipse;
    }

    const cv::Point2f scaled_center{ellipse.center.x / scaling_ratio, ellipse.center.y / scaling_ratio};

    std::optional<cv::RotatedRect> scaled_ellipse;
    if (method == DetectionMethod::Ellipse)
    {
        scaled_ellipse =
            cv::RotatedRect(scaled_center,
                            cv::Size2f(ellipse.size.width / scaling_ratio, ellipse.size.height / scaling_ratio),
                            ellipse.angle);
    }

    return DetectResult{method, scaled_center, scaled_ellipse};
}

} // namespace cheshm::ElSe
