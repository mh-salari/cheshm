// PuReST pupil detector + tracker algorithm body.

#include "PuReST/purest.hpp"

#include "cheshm/canny.hpp"
#include "cheshm/contour_deduplication.hpp"
#include "cheshm/ellipse_sampling.hpp"
#include "cheshm/outline_confidence.hpp"

#include "PuRe/defaults.hpp"
#include "PuRe/pure.hpp"
#include "PuReST/defaults.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <utility>
#include <vector>

namespace cheshm::PuReST
{
namespace
{

using namespace cv;  // NOLINT(google-build-using-namespace)
using namespace std; // NOLINT(google-build-using-namespace)

// Working-frame size cap for the initial PuRe detection (matches the
// PuRe defaults).
constexpr int PURE_BASE_WIDTH = cheshm::PuRe::defaults::BASE_WIDTH;
constexpr int PURE_BASE_HEIGHT = cheshm::PuRe::defaults::BASE_HEIGHT;

inline float majorAxisOf(const RotatedRect& r)
{
    return std::max(r.size.width, r.size.height);
}

inline float minorAxisOf(const RotatedRect& r)
{
    return std::min(r.size.width, r.size.height);
}

inline float circumferenceOf(const RotatedRect& r)
{
    const float a = 0.5f * majorAxisOf(r);
    const float b = 0.5f * minorAxisOf(r);
    return static_cast<float>(CV_PI) * std::abs(3.0f * (a + b) - std::sqrt(10.0f * a * b + 3.0f * (a * a + b * b)));
}

// Ratio of edge pixels found inside a thin band around the fitted
// outline to the outline's circumference. Returns the band edge points
// via the `edgePoints` output param.
float edgeRatioConfidence(const Mat& edgeImage, const RotatedRect& pupil, std::vector<Point>& edgePoints, int band = 5)
{
    if (pupil.size.width <= 0 || pupil.size.height <= 0)
        return 0.0f;

    Mat outlineMask = Mat::zeros(edgeImage.rows, edgeImage.cols, CV_8U);
    ellipse(outlineMask, pupil, Scalar(255), band);

    Mat inBandEdges = edgeImage.clone();
    inBandEdges.setTo(0, 255 - outlineMask);
    findNonZero(inBandEdges, edgePoints);

    return std::min<float>(static_cast<float>(edgePoints.size()) / circumferenceOf(pupil), 1.0f);
}

float aspectRatioConfidence(const RotatedRect& pupil)
{
    return minorAxisOf(pupil) / majorAxisOf(pupil);
}

float angularSpreadConfidence(const std::vector<Point>& points, const cv::Point2f& center)
{
    std::bitset<4> slices;
    for (const Point& p : points)
    {
        if (p.x - center.x < 0)
        {
            slices.set(p.y - center.y < 0 ? 0 : 3);
        }
        else
        {
            slices.set(p.y - center.y < 0 ? 1 : 2);
        }
    }
    return static_cast<float>(slices.count()) / 4.0f;
}

float pupilConfidence(const Mat& frame, const RotatedRect& pupil, const std::vector<Point>& points, int bias)
{
    return 0.34f * cheshm::outline_contrast_confidence(frame, pupil, bias).value_or(0.0f) +
           0.33f * aspectRatioConfidence(pupil) + 0.33f * angularSpreadConfidence(points, pupil.center);
}


void calculateHistogram(const Mat& in, Mat& histogram, int bins)
{
    int channels[] = {0};
    int histSize[] = {bins};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    calcHist(&in, 1, channels, Mat(), histogram, 1, histSize, ranges, true, false);
}

void getThresholds(const Mat& input,
                   const Mat& histogram,
                   const RotatedRect& pupil,
                   int bias,
                   const Mat& open_kernel,
                   const Mat& dilate_kernel,
                   int& lowTh,
                   int& highTh,
                   Mat& bright,
                   Mat& dark)
{
    int th;
    float acc;
    float area;

    acc = 0;
    area = 0.05f * input.rows * input.cols;
    for (th = histogram.rows - 1; th > 0; --th)
    {
        acc += histogram.ptr<float>(th)[0];
        if (acc > area)
            break;
    }
    highTh = th;

    acc = 0;
    area = static_cast<float>(CV_PI) * 0.5f * pupil.size.width * 0.5f * pupil.size.height;
    for (th = 0; th < histogram.rows; ++th)
    {
        acc += histogram.ptr<float>(th)[0];
        if (acc > area)
            break;
    }
    lowTh = th;

    highTh -= bias;

    inRange(input, highTh, 256, bright);
    dilate(bright, bright, open_kernel);

    inRange(input, 0, lowTh, dark);
    dilate(dark, dark, dilate_kernel);
    erode(dark, dark, open_kernel);
}

struct GreedyCandidate
{
    float maxGap;
    std::vector<Point> points;
    std::vector<Point> hull;

    explicit GreedyCandidate(const std::vector<Point>& pts)
        : points(pts)
    {
        convexHull(pts, hull);
        maxGap = 0;
        for (auto p1 = hull.begin(); p1 != hull.end(); ++p1)
        {
            for (auto p2 = p1 + 1; p2 != hull.end(); ++p2)
            {
                const float gap = static_cast<float>(norm(*p2 - *p1));
                if (gap > maxGap)
                    maxGap = gap;
            }
        }
    }
};

void generateCombinations(const std::vector<GreedyCandidate>& seeds,
                          std::vector<GreedyCandidate>& candidates,
                          int length)
{
    if (length > static_cast<int>(seeds.size()))
        return;

    std::vector<bool> v(seeds.size());
    std::fill(v.end() - length, v.end(), true);
    do
    {
        std::vector<Point> points;
        for (std::size_t i = 0; i < seeds.size(); ++i)
        {
            if (v[i])
            {
                const auto& hull = seeds[i].hull;
                points.insert(points.end(), hull.begin(), hull.end());
            }
        }
        candidates.emplace_back(GreedyCandidate(points));
    } while (std::next_permutation(v.begin(), v.end()));
}

// Greedy-search tracker: collect dark-region contour candidates near
// the previous pupil, hull them, generate combination hulls, fit
// ellipses, score by outline-contrast confidence. Returns the
// highest-scoring fit when its confidence clears MIN_GREEDY_CONFIDENCE.
bool greedySearch(const Mat& greedyDetectorEdges,
                  const Mat& input,
                  const RotatedRect& basePupil,
                  const Mat& dark,
                  const Mat& bright,
                  float localMinPupilDiameterPx,
                  RotatedRect& pupil,
                  float& pupilConf)
{
    std::vector<Vec4i> hierarchy;
    std::vector<std::vector<Point>> curves;
    findContours(greedyDetectorEdges, curves, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

    for (auto c = curves.begin(); c != curves.end();)
    {
        if (c->size() < 5)
            c = curves.erase(c);
        else
            ++c;
    }

    for (auto c = curves.begin(); c != curves.end();)
    {
        std::vector<Point> ac;
        approxPolyDP(*c, ac, 1.5, false);
        if (ac.size() > 3)
        {
            ++c;
        }
        else
        {
            c = curves.erase(c);
        }
    }

    cheshm::deduplicate_contours_by_first_point(curves, greedyDetectorEdges.cols);

    std::vector<GreedyCandidate> candidates;
    const float baseMajor = majorAxisOf(basePupil);
    for (std::size_t i = 0; i < curves.size(); ++i)
    {
        GreedyCandidate c(curves[i]);
        if (c.maxGap > 1.25f * baseMajor)
            continue;

        int good = 0, regular = 0, bad = 0;
        for (const Point& p : c.points)
        {
            if (dark.ptr<uchar>(p.y)[p.x] > 0)
            {
                ++good;
            }
            else if (bright.ptr<uchar>(p.y)[p.x] > 0)
            {
                ++bad;
            }
            else
            {
                ++regular;
            }
        }
        if (good > bad && good > regular)
            candidates.push_back(std::move(c));
    }

    if (candidates.empty())
        return false;

    std::sort(candidates.begin(),
              candidates.end(),
              [](const GreedyCandidate& a, const GreedyCandidate& b) { return a.maxGap > b.maxGap; });
    while (candidates.size() > 5)
        candidates.pop_back();

    std::vector<GreedyCandidate> combined;
    for (int length = 1; length <= static_cast<int>(candidates.size()); ++length)
        generateCombinations(candidates, combined, length);
    candidates.insert(
        candidates.end(), std::make_move_iterator(combined.begin()), std::make_move_iterator(combined.end()));

    constexpr float MIN_CURVATURE_RATIO = 0.198912f;
    RotatedRect bestPupil;
    float bestConf = 0.0f;
    for (const auto& c : candidates)
    {
        if (c.hull.size() < 5)
            continue;
        const RotatedRect p = fitEllipse(c.hull);
        if (majorAxisOf(p) < localMinPupilDiameterPx)
            continue;
        const float aspect = minorAxisOf(p) / majorAxisOf(p);
        if (aspect < MIN_CURVATURE_RATIO)
            continue;
        const float conf = cheshm::outline_contrast_confidence(input, p, defaults::THRESHOLD_BIAS).value_or(0.0f);
        if (conf > bestConf)
        {
            bestConf = conf;
            bestPupil = p;
        }
    }

    if (bestConf > defaults::MIN_GREEDY_CONFIDENCE && bestPupil.size.width > 0 && bestPupil.size.height > 0 &&
        bestPupil.center.x > 0 && bestPupil.center.y > 0)
    {
        pupil = bestPupil;
        pupilConf = bestConf;
        return true;
    }
    return false;
}

// Outline-tracker: refit an ellipse to edges found in a band around
// the previous pupil's outline. Returns the refined fit when:
//  (1) enough edges are found in the band,
//  (2) the band edge-ratio confidence clears the floor on both fits,
//  (3) the fit's major axis hasn't blown up vs the original seed.
bool trackOutline(const Mat& outlineTrackerEdges,
                  const Mat& input,
                  const RotatedRect& basePupil,
                  float localScalingRatio,
                  RotatedRect& outlineSeedPupil,
                  bool& outlineSeedValid,
                  int outline_bias,
                  RotatedRect& pupil,
                  float& pupilConf,
                  float minOutlineConfidence = defaults::MIN_OUTLINE_CONFIDENCE)
{
    std::vector<Point> edges;

    if (!outlineSeedValid)
    {
        outlineSeedPupil = basePupil;
        // Outline seed pupil is stored in full-frame coords; scale it up.
        outlineSeedPupil.center.x *= 1.0f / localScalingRatio;
        outlineSeedPupil.center.y *= 1.0f / localScalingRatio;
        outlineSeedPupil.size.width *= 1.0f / localScalingRatio;
        outlineSeedPupil.size.height *= 1.0f / localScalingRatio;
        outlineSeedValid = true;
    }

    float edgeRatio = edgeRatioConfidence(outlineTrackerEdges, basePupil, edges);

    if (edges.size() > 5 && edgeRatio > minOutlineConfidence)
    {
        RotatedRect outlineTracker = fitEllipse(edges);
        edgeRatio = edgeRatioConfidence(outlineTrackerEdges, outlineTracker, edges);

        if (edges.size() > 5 && edgeRatio > minOutlineConfidence)
        {
            outlineTracker = fitEllipse(edges);
            const float conf = pupilConfidence(input, outlineTracker, edges, outline_bias);

            const float majorRatio =
                ((1.0f / localScalingRatio) * majorAxisOf(outlineTracker)) / majorAxisOf(outlineSeedPupil);

            if (majorRatio < defaults::MAX_MAJOR_AXIS_RATIO && outlineTracker.size.width > 0 &&
                outlineTracker.size.height > 0 && outlineTracker.center.x > 0 && outlineTracker.center.y > 0)
            {
                pupil = outlineTracker;
                pupilConf = conf;
                return true;
            }
        }
    }

    outlineSeedValid = false;
    return false;
}

bool runTracking(const Mat& frame,
                 const RotatedRect& previous_pupil,
                 float min_pupil_diameter_mm,
                 float max_pupil_diameter_mm,
                 float canthi_distance_mm,
                 int outline_bias,
                 const Mat& open_kernel,
                 const Mat& dilate_kernel,
                 RotatedRect& outline_seed_pupil,
                 bool& outline_seed_valid,
                 RotatedRect& result,
                 float& result_conf)
{
    // Compute the tracking ROI: 2*majorAxis box around the previous pupil.
    const Rect frameRect{0, 0, frame.cols, frame.rows};
    const double trackingHalfSide = std::max(previous_pupil.size.width, previous_pupil.size.height);
    const Point2f delta{static_cast<float>(trackingHalfSide), static_cast<float>(trackingHalfSide)};
    Rect trackingRect = Rect(previous_pupil.center - delta, previous_pupil.center + delta) & frameRect;
    if (trackingRect.width < 10 || trackingRect.height < 10)
        return false;

    // Initial scaling matches PuRe's base size on the full frame.
    const float rw = static_cast<float>(PURE_BASE_WIDTH) / static_cast<float>(frame.cols);
    const float rh = static_cast<float>(PURE_BASE_HEIGHT) / static_cast<float>(frame.rows);
    float localScalingRatio = std::min<float>(std::min<float>(rw, rh), 1.0f);

    // If the resulting tracking rect would scale up too large, reduce
    // the scaling ratio to bound runtime.
    Size scaledSize = trackingRect.size();
    scaledSize.width = static_cast<int>(scaledSize.width * localScalingRatio);
    scaledSize.height = static_cast<int>(scaledSize.height * localScalingRatio);
    if (scaledSize.width > defaults::MAX_SCALED_REGION || scaledSize.height > defaults::MAX_SCALED_REGION)
    {
        const float r =
            std::min<float>(static_cast<float>(defaults::MAX_SCALED_REGION) / static_cast<float>(trackingRect.width),
                            static_cast<float>(defaults::MAX_SCALED_REGION) / static_cast<float>(trackingRect.height));
        localScalingRatio = r;
    }

    // Pupil-pixel bounds derived from canthi-distance assumptions.
    const int scaledRows = static_cast<int>(localScalingRatio * frame.rows);
    const int scaledCols = static_cast<int>(localScalingRatio * frame.cols);
    const float d = std::sqrt(static_cast<float>(scaledRows * scaledRows + scaledCols * scaledCols));
    const int maxPupilDiameterPx = static_cast<int>(d * (max_pupil_diameter_mm / canthi_distance_mm));
    const int minPupilDiameterPx = static_cast<int>((2.0f * d / 3.0f) * (min_pupil_diameter_mm / canthi_distance_mm));
    (void)maxPupilDiameterPx; // currently unused in tracker path; kept for parity with reference.

    // Crop + scale.
    Mat input;
    resize(frame(trackingRect), input, Size(), localScalingRatio, localScalingRatio, INTER_LINEAR);

    // Canny working buffers.
    Mat dx = Mat::zeros(input.size(), CV_32F);
    Mat dy = Mat::zeros(input.size(), CV_32F);
    Mat magnitude = Mat::zeros(input.size(), CV_32F);
    Mat edgeType = Mat::zeros(input.size(), CV_8U);
    Mat edge = Mat::zeros(input.size(), CV_8U);

    // Previous pupil in the working (cropped + scaled) coordinate system.
    RotatedRect basePupil = previous_pupil;
    basePupil.center.x -= trackingRect.x;
    basePupil.center.y -= trackingRect.y;
    basePupil.center.x *= localScalingRatio;
    basePupil.center.y *= localScalingRatio;
    basePupil.size.width *= localScalingRatio;
    basePupil.size.height *= localScalingRatio;

    Mat histogram;
    calculateHistogram(input, histogram, 256);

    int lowTh, highTh;
    Mat bright, dark;
    getThresholds(input,
                  histogram,
                  basePupil,
                  defaults::THRESHOLD_BIAS,
                  open_kernel,
                  dilate_kernel,
                  lowTh,
                  highTh,
                  bright,
                  dark);

    Mat detectedEdges = cheshm::canny(input,
                                      dx,
                                      dy,
                                      magnitude,
                                      edgeType,
                                      edge,
                                      cheshm::PuRe::defaults::CANNY_BINS,
                                      cheshm::PuRe::defaults::CANNY_NON_EDGE_RATIO,
                                      cheshm::PuRe::defaults::CANNY_LOW_HIGH_RATIO);
    cheshm::filter_edges(detectedEdges);

    Mat outlineTrackerEdges = detectedEdges.clone();
    outlineTrackerEdges.setTo(0, bright);
    outlineTrackerEdges.setTo(0, 255 - dark);

    RotatedRect outlinePupil;
    float outlineConf = 0.0f;
    if (trackOutline(outlineTrackerEdges,
                     input,
                     basePupil,
                     localScalingRatio,
                     outline_seed_pupil,
                     outline_seed_valid,
                     outline_bias,
                     outlinePupil,
                     outlineConf))
    {
        // Scale back to full-frame coordinates.
        outlinePupil.center.x /= localScalingRatio;
        outlinePupil.center.y /= localScalingRatio;
        outlinePupil.size.width /= localScalingRatio;
        outlinePupil.size.height /= localScalingRatio;
        outlinePupil.center.x += trackingRect.x;
        outlinePupil.center.y += trackingRect.y;
        result = outlinePupil;
        result_conf = outlineConf;
        return true;
    }

    RotatedRect greedyPupil;
    float greedyConf = 0.0f;
    Mat greedyDetectorEdges = detectedEdges.clone();
    if (greedySearch(greedyDetectorEdges,
                     input,
                     basePupil,
                     dark,
                     bright,
                     localScalingRatio * minPupilDiameterPx,
                     greedyPupil,
                     greedyConf))
    {
        greedyPupil.center.x /= localScalingRatio;
        greedyPupil.center.y /= localScalingRatio;
        greedyPupil.size.width /= localScalingRatio;
        greedyPupil.size.height /= localScalingRatio;
        greedyPupil.center.x += trackingRect.x;
        greedyPupil.center.y += trackingRect.y;
        result = greedyPupil;
        result_conf = greedyConf;
        return true;
    }

    return false;
}

} // namespace

Tracker::Tracker(float min_pupil_diameter_mm, float max_pupil_diameter_mm, float canthi_distance_mm, int outline_bias)
    : min_pupil_diameter_mm_(min_pupil_diameter_mm),
      max_pupil_diameter_mm_(max_pupil_diameter_mm),
      canthi_distance_mm_(canthi_distance_mm),
      outline_bias_(outline_bias),
      has_previous_(false),
      previous_confidence_(0.0f),
      outline_seed_valid_(false),
      open_kernel_(cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7})),
      dilate_kernel_(cv::getStructuringElement(cv::MORPH_ELLIPSE, {15, 15}))
{
}

void Tracker::reset()
{
    has_previous_ = false;
    previous_confidence_ = 0.0f;
    outline_seed_valid_ = false;
    previous_pupil_ = cv::RotatedRect();
    outline_seed_pupil_ = cv::RotatedRect();
}

std::optional<DetectResult> Tracker::detect(const cv::Mat& frame)
{
    if (frame.empty())
        return std::nullopt;

    cv::RotatedRect ellipse;
    float confidence = 0.0f;
    bool ok = false;

    if (has_previous_)
    {
        ok = runTracking(frame,
                         previous_pupil_,
                         min_pupil_diameter_mm_,
                         max_pupil_diameter_mm_,
                         canthi_distance_mm_,
                         outline_bias_,
                         open_kernel_,
                         dilate_kernel_,
                         outline_seed_pupil_,
                         outline_seed_valid_,
                         ellipse,
                         confidence);
    }

    if (!ok)
    {
        // Fall back to full PuRe detection.
        const auto fallback = cheshm::PuRe::detect(
            frame, min_pupil_diameter_mm_, max_pupil_diameter_mm_, canthi_distance_mm_, outline_bias_);
        if (!fallback)
        {
            has_previous_ = false;
            outline_seed_valid_ = false;
            return std::nullopt;
        }
        ellipse = fallback->ellipse;
        confidence = fallback->confidence;
        outline_seed_valid_ = false;
    }

    previous_pupil_ = ellipse;
    previous_confidence_ = confidence;
    has_previous_ = true;
    return DetectResult{ellipse, confidence};
}

} // namespace cheshm::PuReST
