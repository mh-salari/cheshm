#include "cheshm/align/rigid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <poolstl/poolstl.hpp>
#include <vector>

namespace cheshm::align
{

namespace
{

constexpr double kDegPerRad = 180.0 / CV_PI;

void linspace(double lo, double hi, int n, std::vector<double>& out)
{
    out.clear();
    out.reserve(static_cast<std::size_t>(n));
    if (n <= 1)
    {
        out.push_back(lo);
        return;
    }
    const double step = (hi - lo) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i)
        out.push_back(lo + static_cast<double>(i) * step);
}

int banker_round(double x) noexcept
{
    return static_cast<int>(std::lrint(x));
}

struct ShiftSearchResult
{
    double dx;
    double dy;
    double score;
};

ShiftSearchResult search_shifts(const std::vector<double>& ref_vals,
                                const cv::Mat& img,
                                const std::vector<double>& mask_rows,
                                const std::vector<double>& mask_cols,
                                const std::vector<double>& dxs,
                                const std::vector<double>& dys)
{
    const int h = img.rows;
    const int w = img.cols;
    const std::size_t n_dx = dxs.size();
    const std::size_t n_dy = dys.size();
    const std::size_t n_combos = n_dx * n_dy;
    const std::size_t n = ref_vals.size();

    std::vector<double> scores(n_combos, std::numeric_limits<double>::max());

    auto begin = poolstl::iota_iter<std::size_t>(0);
    auto end = poolstl::iota_iter<std::size_t>(n_combos);
    std::for_each(poolstl::par,
                  begin,
                  end,
                  [&](std::size_t idx)
                  {
                      const std::size_t di = idx / n_dy;
                      const std::size_t dj = idx % n_dy;
                      const double dx = dxs[di];
                      const double dy = dys[dj];
                      double total = 0.0;
                      std::size_t count = 0;
                      for (std::size_t k = 0; k < n; ++k)
                      {
                          const double fy = mask_rows[k] - dy;
                          const double fx = mask_cols[k] - dx;
                          const int x0 = static_cast<int>(std::floor(fx));
                          const int y0 = static_cast<int>(std::floor(fy));
                          const int x1 = x0 + 1;
                          const int y1 = y0 + 1;
                          if (x0 < 0 || y0 < 0 || x1 >= w || y1 >= h)
                              continue;
                          const double wx = fx - static_cast<double>(x0);
                          const double wy = fy - static_cast<double>(y0);
                          const double v00 = img.at<double>(y0, x0);
                          const double v01 = img.at<double>(y0, x1);
                          const double v10 = img.at<double>(y1, x0);
                          const double v11 = img.at<double>(y1, x1);
                          const double val = v00 * (1.0 - wx) * (1.0 - wy) + v01 * wx * (1.0 - wy) +
                                             v10 * (1.0 - wx) * wy + v11 * wx * wy;
                          total += std::abs(ref_vals[k] - val);
                          ++count;
                      }
                      if (count > 0)
                          scores[idx] = total / static_cast<double>(count);
                  });

    const auto best = std::min_element(scores.begin(), scores.end());
    const std::size_t best_idx = static_cast<std::size_t>(std::distance(scores.begin(), best));
    return {dxs[best_idx / n_dy], dys[best_idx % n_dy], *best};
}

struct GridSearchResult
{
    cv::Vec3d params;
    double score;
};

GridSearchResult grid_search(const cv::Mat& ref_f,
                             const cv::Mat& img_mov,
                             const cv::Mat& mask,
                             const std::vector<double>& thetas,
                             const std::vector<double>& dxs,
                             const std::vector<double>& dys,
                             cv::Point2d center)
{
    const int h = ref_f.rows;
    const int w = ref_f.cols;

    std::vector<double> mask_rows;
    std::vector<double> mask_cols;
    std::vector<double> ref_vals;
    const std::size_t nz = static_cast<std::size_t>(cv::countNonZero(mask));
    mask_rows.reserve(nz);
    mask_cols.reserve(nz);
    ref_vals.reserve(nz);
    for (int y = 0; y < h; ++y)
    {
        const std::uint8_t* m = mask.ptr<std::uint8_t>(y);
        const float* r = ref_f.ptr<float>(y);
        for (int x = 0; x < w; ++x)
        {
            if (m[x])
            {
                mask_rows.push_back(static_cast<double>(y));
                mask_cols.push_back(static_cast<double>(x));
                ref_vals.push_back(static_cast<double>(r[x]));
            }
        }
    }

    GridSearchResult best{cv::Vec3d{0.0, 0.0, 0.0}, std::numeric_limits<double>::infinity()};

    cv::Mat rotated;
    cv::Mat rotated_f;
    for (double theta : thetas)
    {
        const cv::Mat R = cv::getRotationMatrix2D(center, theta, 1.0);
        cv::warpAffine(img_mov, rotated, R, cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        rotated.convertTo(rotated_f, CV_64F);
        const auto shift = search_shifts(ref_vals, rotated_f, mask_rows, mask_cols, dxs, dys);
        if (shift.score < best.score)
        {
            best.score = shift.score;
            best.params = cv::Vec3d{shift.dx, shift.dy, theta};
        }
    }
    return best;
}

} // namespace

cv::Mat make_iris_mask(cv::Size sz,
                       cv::Point2d limbus_center,
                       double limbus_r,
                       double pupil_r,
                       double exclude_top,
                       double exclude_bottom,
                       double inner_margin)
{
    cv::Mat mask = cv::Mat::zeros(sz, CV_8U);
    const double inner = pupil_r + inner_margin;
    const double outer = limbus_r + defaults::OUTER_MARGIN;
    const double inner_sq = inner * inner;
    const double outer_sq = outer * outer;
    for (int y = 0; y < sz.height; ++y)
    {
        std::uint8_t* row = mask.ptr<std::uint8_t>(y);
        const double dyy = static_cast<double>(y) - limbus_center.y;
        for (int x = 0; x < sz.width; ++x)
        {
            const double dxx = static_cast<double>(x) - limbus_center.x;
            const double d_sq = dxx * dxx + dyy * dyy;
            if (d_sq < inner_sq || d_sq > outer_sq)
                continue;
            const double angle = std::atan2(dyy, dxx) * kDegPerRad;
            const bool in_top = (angle >= -90.0 - exclude_top) && (angle <= -90.0 + exclude_top);
            const bool in_bottom = (angle >= 90.0 - exclude_bottom) && (angle <= 90.0 + exclude_bottom);
            if (!in_top && !in_bottom)
                row[x] = 255;
        }
    }
    return mask;
}

cv::Mat make_barrel_mask(cv::Size sz,
                         cv::Point2d limbus_center,
                         double limbus_r,
                         double pupil_r,
                         double exclude_top,
                         double exclude_bottom,
                         double inner_margin)
{
    const cv::Mat ring =
        make_iris_mask(sz, limbus_center, limbus_r, pupil_r, exclude_top, exclude_bottom, inner_margin);
    cv::Mat mask = ring.clone();
    for (int y = 0; y < ring.rows; ++y)
    {
        const std::uint8_t* row = ring.ptr<std::uint8_t>(y);
        int first = -1;
        int last = -1;
        for (int x = 0; x < ring.cols; ++x)
        {
            if (row[x])
            {
                if (first < 0)
                    first = x;
                last = x;
            }
        }
        if (first >= 0 && last > first)
        {
            std::uint8_t* m_row = mask.ptr<std::uint8_t>(y);
            std::fill(m_row + first, m_row + last + 1, std::uint8_t{255});
        }
    }
    return mask;
}

cv::Vec3d align_by_translation(cv::Point2d ref_point, cv::Point2d mov_point)
{
    return cv::Vec3d{ref_point.x - mov_point.x, ref_point.y - mov_point.y, 0.0};
}

cv::Mat apply_transform(const cv::Mat& img, cv::Vec3d params, std::optional<cv::Point2d> center_opt)
{
    const cv::Point2d center = center_opt.value_or(cv::Point2d(img.cols / 2.0, img.rows / 2.0));
    cv::Mat M = cv::getRotationMatrix2D(center, params[2], 1.0);
    M.at<double>(0, 2) += params[0];
    M.at<double>(1, 2) += params[1];
    cv::Mat warped;
    cv::warpAffine(img, warped, M, img.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return warped;
}

std::pair<cv::Vec3d, double> align_by_min_diff(const cv::Mat& img_ref,
                                               const cv::Mat& img_mov,
                                               const cv::Mat& mask,
                                               int dx_lo,
                                               int dx_hi,
                                               int dy_lo,
                                               int dy_hi,
                                               double rot_start,
                                               double rot_end,
                                               double rot_step,
                                               std::optional<cv::Point2d> rotation_center_opt)
{
    const cv::Point2d center = rotation_center_opt.value_or(cv::Point2d(img_ref.cols / 2.0, img_ref.rows / 2.0));
    cv::Mat ref_f;
    img_ref.convertTo(ref_f, CV_32F);

    const int n_rot = static_cast<int>(std::round((rot_end - rot_start) / rot_step)) + 1;
    std::vector<double> thetas;
    linspace(rot_start, rot_end, n_rot, thetas);

    std::vector<double> coarse_dxs;
    std::vector<double> coarse_dys;
    coarse_dxs.reserve(static_cast<std::size_t>(dx_hi - dx_lo));
    coarse_dys.reserve(static_cast<std::size_t>(dy_hi - dy_lo));
    for (int i = dx_lo; i < dx_hi; ++i)
        coarse_dxs.push_back(static_cast<double>(i));
    for (int i = dy_lo; i < dy_hi; ++i)
        coarse_dys.push_back(static_cast<double>(i));

    const auto coarse = grid_search(ref_f, img_mov, mask, thetas, coarse_dxs, coarse_dys, center);
    const double cdx = coarse.params[0];
    const double cdy = coarse.params[1];
    const double ctheta = coarse.params[2];

    std::vector<double> fine_thetas;
    std::vector<double> fine_dxs;
    std::vector<double> fine_dys;
    linspace(ctheta - defaults::FINE_ROT_HALF, ctheta + defaults::FINE_ROT_HALF, defaults::FINE_ROT_N, fine_thetas);
    linspace(cdx - defaults::FINE_SHIFT_HALF, cdx + defaults::FINE_SHIFT_HALF, defaults::FINE_SHIFT_N, fine_dxs);
    linspace(cdy - defaults::FINE_SHIFT_HALF, cdy + defaults::FINE_SHIFT_HALF, defaults::FINE_SHIFT_N, fine_dys);

    const auto fine = grid_search(ref_f, img_mov, mask, fine_thetas, fine_dxs, fine_dys, center);
    return {fine.params, fine.score};
}

namespace
{

cv::Point2d glint_centroid(const std::vector<cv::Point2d>& glints)
{
    cv::Point2d sum{0.0, 0.0};
    for (const auto& g : glints)
        sum += g;
    const double n = static_cast<double>(glints.size());
    return {sum.x / n, sum.y / n};
}

double median_nn_distance(const std::vector<cv::Point2d>& pts)
{
    if (pts.size() < 2)
        return -1.0;
    std::vector<double> nn;
    nn.reserve(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        double best = std::numeric_limits<double>::max();
        for (std::size_t j = 0; j < pts.size(); ++j)
            if (i != j)
                best = std::min(best, cv::norm(pts[i] - pts[j]));
        nn.push_back(best);
    }
    std::sort(nn.begin(), nn.end());
    const std::size_t m = nn.size();
    return (m % 2 != 0) ? nn[m / 2] : 0.5 * (nn[m / 2 - 1] + nn[m / 2]);
}

struct MatchCandidate
{
    double dist;
    int mov_index;
    int ref_index;
};

// Greedy one-to-one matches (moving -> reference) within ``tol``, nearest first.
std::vector<std::pair<int, int>> greedy_match(const std::vector<cv::Point2d>& shifted,
                                              const std::vector<cv::Point2d>& reference,
                                              double tol)
{
    std::vector<MatchCandidate> candidates;
    for (int i = 0; i < static_cast<int>(shifted.size()); ++i)
        for (int j = 0; j < static_cast<int>(reference.size()); ++j)
        {
            const double dist = cv::norm(shifted[i] - reference[j]);
            if (dist <= tol)
                candidates.push_back({dist, i, j});
        }
    std::sort(candidates.begin(),
              candidates.end(),
              [](const MatchCandidate& a, const MatchCandidate& b) { return a.dist < b.dist; });
    std::vector<char> used_mov(shifted.size(), 0);
    std::vector<char> used_ref(reference.size(), 0);
    std::vector<std::pair<int, int>> pairs;
    for (const auto& c : candidates)
    {
        if (used_mov[c.mov_index] || used_ref[c.ref_index])
            continue;
        used_mov[c.mov_index] = 1;
        used_ref[c.ref_index] = 1;
        pairs.emplace_back(c.mov_index, c.ref_index);
    }
    return pairs;
}

} // namespace

cv::Point2d match_glints(const std::vector<cv::Point2d>& reference,
                         const std::vector<cv::Point2d>& moving,
                         double tol_fraction)
{
    if (reference.empty() || moving.empty())
        return {0.0, 0.0};
    double spacing = median_nn_distance(reference);
    if (spacing < 0.0)
        spacing = median_nn_distance(moving);
    if (spacing < 0.0)
        return glint_centroid(reference) - glint_centroid(moving); // a single point in each set
    const double tol = tol_fraction * spacing;

    int best_count = -1;
    double best_residual = 0.0;
    std::vector<std::pair<int, int>> best_pairs;
    std::vector<cv::Point2d> shifted(moving.size());
    for (const auto& ref_pt : reference)
        for (const auto& mov_pt : moving)
        {
            const cv::Point2d t = ref_pt - mov_pt;
            for (std::size_t k = 0; k < moving.size(); ++k)
                shifted[k] = moving[k] + t;
            const std::vector<std::pair<int, int>> pairs = greedy_match(shifted, reference, tol);
            double residual = 0.0;
            for (const auto& [mov_index, ref_index] : pairs)
                residual += cv::norm(shifted[mov_index] - reference[ref_index]);
            const int count = static_cast<int>(pairs.size());
            if (count > best_count || (count == best_count && residual < best_residual))
            {
                best_count = count;
                best_residual = residual;
                best_pairs = pairs;
            }
        }
    if (best_pairs.empty())
        return glint_centroid(reference) - glint_centroid(moving);
    cv::Point2d sum{0.0, 0.0};
    for (const auto& [mov_index, ref_index] : best_pairs)
        sum += reference[ref_index] - moving[mov_index];
    return sum * (1.0 / static_cast<double>(best_pairs.size()));
}

AlignResult align_eye_images(const cv::Mat& ref_img,
                             const cv::Mat& tgt_img,
                             const EyeDetection& ref_det,
                             const EyeDetection& tgt_det,
                             Step1Anchor step1,
                             bool step2,
                             double exclude_top,
                             double exclude_bottom,
                             double inner_margin)
{
    AlignResult out;

    if (step1 == Step1Anchor::None && !step2)
    {
        out.aligned = tgt_img.clone();
        return out;
    }

    cv::Vec3d p1{0.0, 0.0, 0.0};
    cv::Mat warped;

    std::optional<cv::Point2d> step2_center;
    if (step2)
    {
        const double cx = (ref_det.limbus_center.x + tgt_det.limbus_center.x) / 2.0;
        const double cy = (ref_det.limbus_center.y + tgt_det.limbus_center.y) / 2.0;
        step2_center = cv::Point2d{static_cast<double>(banker_round(cx)), static_cast<double>(banker_round(cy))};
        out.rotation_center = cv::Point2i{banker_round(cx), banker_round(cy)};
    }

    if (step1 == Step1Anchor::None)
    {
        warped = tgt_img.clone();
    }
    else
    {
        if (step1 == Step1Anchor::Glint)
        {
            const cv::Point2d t = match_glints(ref_det.glints, tgt_det.glints);
            p1 = cv::Vec3d{t.x, t.y, 0.0};
        }
        else
        {
            p1 = align_by_translation(ref_det.pupil_center, tgt_det.pupil_center);
        }
        out.step1_translation = cv::Point2d{p1[0], p1[1]};
        warped = apply_transform(tgt_img, p1, step2_center);
    }

    if (step2)
    {
        const double iris_r = static_cast<double>(banker_round((ref_det.limbus_radius + tgt_det.limbus_radius) / 2.0));
        const double pupil_r = static_cast<double>(banker_round(std::max(ref_det.pupil_radius, tgt_det.pupil_radius)));
        const cv::Mat barrel = make_barrel_mask(
            ref_img.size(), *step2_center, iris_r, pupil_r, exclude_top, exclude_bottom, inner_margin);
        const auto [p2, _score] = align_by_min_diff(ref_img,
                                                    warped,
                                                    barrel,
                                                    defaults::DX_LO,
                                                    defaults::DX_HI,
                                                    defaults::DY_LO,
                                                    defaults::DY_HI,
                                                    defaults::ROT_START,
                                                    defaults::ROT_END,
                                                    defaults::ROT_STEP,
                                                    step2_center);
        out.aligned = apply_transform(warped, p2, step2_center);
        out.step2_transform = p2;
    }
    else
    {
        out.aligned = warped;
    }

    return out;
}

} // namespace cheshm::align
