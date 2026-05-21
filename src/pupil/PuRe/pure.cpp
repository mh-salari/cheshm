// PuRe pupil detector algorithm body.

#include "cheshm/pupil/PuRe/pure.hpp"

#include "cheshm/helpers/edges/canny.hpp"
#include "cheshm/helpers/edges/edge_filter.hpp"
#include "cheshm/helpers/ellipses/ellipse_sampling.hpp"
#include "cheshm/helpers/ellipses/outline_confidence.hpp"
#include "cheshm/helpers/image/normalise.hpp"
#include "cheshm/helpers/shape/contour_deduplication.hpp"
#include "cheshm/pupil/PuRe/defaults.hpp"

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

namespace cheshm::PuRe
{
namespace
{

using namespace cv;  // NOLINT(google-build-using-namespace)
using namespace std; // NOLINT(google-build-using-namespace)

// (1 - cos(22.5°)) / sin(22.5°) — minimum minor/major axis ratio that
// keeps a candidate above the "too flat to be a pupil" gate.
constexpr float MIN_CURVATURE_RATIO = 0.198912f;

struct PupilCandidate
{
    std::vector<Point> points;
    RotatedRect pointsMinAreaRect;
    RotatedRect outline;
    Rect pointsBoundingBox;
    Rect combinationRegion;
    Rect boundaries;
    cv::Point2f v[4];
    cv::Point2f mp;
    float minorAxis = 0.0f;
    float majorAxis = 0.0f;
    float aspectRatio = 0.0f;
    float outlineContrast = 0.0f;
    float anchorDistribution = 0.0f;
    float score = 0.0f;
    std::bitset<4> anchorPointSlices;

    enum
    {
        Q0 = 0,
        Q1 = 1,
        Q2 = 2,
        Q3 = 3
    };

    explicit PupilCandidate(std::vector<Point> pts)
        : points(std::move(pts))
    {
    }

    bool operator<(const PupilCandidate& c) const
    {
        return score < c.score;
    }

    static float ratio(float a, float b)
    {
        std::pair<float, float> sorted = std::minmax(a, b);
        return sorted.first / sorted.second;
    }

    bool isValid(const Mat& intensityImage, int minPupilDiameterPx, int maxPupilDiameterPx, int bias);
    bool fastValidityCheck(int maxPupilDiameterPx);
    bool validityCheck(const Mat& intensityImage, int bias);
    bool validateAnchorDistribution();
    bool validateOutlineContrast(const Mat& intensityImage, int bias);

    void updateScore()
    {
        score = 0.33f * aspectRatio + 0.33f * anchorDistribution + 0.34f * outlineContrast;
    }
};

bool PupilCandidate::isValid(const Mat& intensityImage, int minPupilDiameterPx, int maxPupilDiameterPx, int bias)
{
    if (points.size() < 5)
        return false;

    float maxGap = 0;
    for (auto p1 = points.begin(); p1 != points.end(); ++p1)
    {
        for (auto p2 = p1 + 1; p2 != points.end(); ++p2)
        {
            const float gap = static_cast<float>(norm(*p2 - *p1));
            if (gap > maxGap)
                maxGap = gap;
        }
    }

    if (maxGap >= maxPupilDiameterPx)
        return false;
    if (maxGap <= minPupilDiameterPx)
        return false;

    outline = fitEllipse(points);
    boundaries = {0, 0, intensityImage.cols, intensityImage.rows};

    if (!boundaries.contains(outline.center))
        return false;

    if (!fastValidityCheck(maxPupilDiameterPx))
        return false;

    pointsMinAreaRect = minAreaRect(points);
    if (ratio(pointsMinAreaRect.size.width, pointsMinAreaRect.size.height) < MIN_CURVATURE_RATIO)
        return false;

    if (!validityCheck(intensityImage, bias))
        return false;

    updateScore();
    return true;
}

bool PupilCandidate::fastValidityCheck(int maxPupilDiameterPx)
{
    const std::pair<float, float> axis = std::minmax(outline.size.width, outline.size.height);
    minorAxis = axis.first;
    majorAxis = axis.second;
    aspectRatio = minorAxis / majorAxis;

    if (aspectRatio < MIN_CURVATURE_RATIO)
        return false;
    if (majorAxis > maxPupilDiameterPx)
        return false;

    combinationRegion = boundingRect(points);
    combinationRegion.width = std::max<int>(combinationRegion.width, combinationRegion.height);
    combinationRegion.height = combinationRegion.width;

    return true;
}

bool PupilCandidate::validateOutlineContrast(const Mat& intensityImage, int bias)
{
    const auto score = cheshm::outline_contrast_confidence(intensityImage, outline, bias);
    if (!score)
        return false;
    outlineContrast = *score;
    return true;
}

bool PupilCandidate::validateAnchorDistribution()
{
    anchorPointSlices.reset();
    for (const Point& p : points)
    {
        if (p.x - outline.center.x < 0)
        {
            if (p.y - outline.center.y < 0)
                anchorPointSlices.set(Q0);
            else
                anchorPointSlices.set(Q3);
        }
        else
        {
            if (p.y - outline.center.y < 0)
                anchorPointSlices.set(Q1);
            else
                anchorPointSlices.set(Q2);
        }
    }
    anchorDistribution = static_cast<float>(anchorPointSlices.count()) / static_cast<float>(anchorPointSlices.size());
    return true;
}

bool PupilCandidate::validityCheck(const Mat& intensityImage, int bias)
{
    mp = std::accumulate(points.begin(), points.end(), Point(0, 0));
    mp.x = std::roundf(mp.x / points.size());
    mp.y = std::roundf(mp.y / points.size());

    outline.points(v);
    const std::vector<cv::Point2f> pv(v, v + 4);
    if (cv::pointPolygonTest(pv, mp, false) <= 0)
        return false;

    if (!validateAnchorDistribution())
        return false;

    if (!validateOutlineContrast(intensityImage, bias))
        return false;

    return true;
}


void findPupilEdgeCandidates(const Mat& intensityImage,
                             Mat& edge,
                             std::vector<PupilCandidate>& candidates,
                             int minPupilDiameterPx,
                             int maxPupilDiameterPx,
                             int outlineBias)
{
    std::vector<Vec4i> hierarchy;
    std::vector<std::vector<Point>> curves;
    findContours(edge, curves, hierarchy, RETR_LIST, CHAIN_APPROX_TC89_KCOS);

    cheshm::deduplicate_contours_by_first_point(curves, edge.cols);

    for (std::size_t i = curves.size(); i-- > 0;)
    {
        PupilCandidate candidate(curves[i]);
        if (candidate.isValid(intensityImage, minPupilDiameterPx, maxPupilDiameterPx, outlineBias))
            candidates.push_back(candidate);
    }
}

void combineEdgeCandidates(const Mat& intensityImage,
                           std::vector<PupilCandidate>& candidates,
                           int minPupilDiameterPx,
                           int maxPupilDiameterPx,
                           int outlineBias)
{
    if (candidates.size() <= 1)
        return;

    std::vector<PupilCandidate> mergedCandidates;
    for (auto pc = candidates.begin(); pc != candidates.end(); ++pc)
    {
        for (auto pc2 = pc + 1; pc2 != candidates.end(); ++pc2)
        {
            const Rect intersection = pc->combinationRegion & pc2->combinationRegion;
            if (intersection.area() < 1)
                continue;

            if (intersection.area() >= std::min<int>(pc->combinationRegion.area(), pc2->combinationRegion.area()))
                continue;

            std::vector<Point> mergedPoints = pc->points;
            mergedPoints.insert(mergedPoints.end(), pc2->points.begin(), pc2->points.end());
            PupilCandidate candidate(mergedPoints);
            if (!candidate.isValid(intensityImage, minPupilDiameterPx, maxPupilDiameterPx, outlineBias))
                continue;
            if (candidate.outlineContrast < pc->outlineContrast || candidate.outlineContrast < pc2->outlineContrast)
                continue;
            mergedCandidates.push_back(candidate);
        }
    }
    candidates.insert(candidates.end(), mergedCandidates.begin(), mergedCandidates.end());
}

void searchInnerCandidates(const std::vector<PupilCandidate>& candidates, PupilCandidate& selected)
{
    if (candidates.size() <= 1)
        return;

    const float searchRadius = 0.5f * selected.majorAxis;
    std::vector<PupilCandidate> insiders;
    for (const auto& pc : candidates)
    {
        if (searchRadius < pc.majorAxis)
            continue;
        if (norm(selected.outline.center - pc.outline.center) > searchRadius)
            continue;
        if (pc.outlineContrast < 0.75f)
            continue;
        insiders.push_back(pc);
    }
    if (insiders.empty())
        return;

    std::sort(insiders.begin(), insiders.end());
    selected = insiders.back();
}

} // namespace

std::optional<DetectResult> detect(const cv::Mat& frame,
                                   float min_pupil_diameter_mm,
                                   float max_pupil_diameter_mm,
                                   float canthi_distance_mm,
                                   int outline_bias)
{
    if (frame.empty())
        return std::nullopt;

    // Downscale to working size.
    const float rw = static_cast<float>(defaults::BASE_WIDTH) / static_cast<float>(frame.cols);
    const float rh = static_cast<float>(defaults::BASE_HEIGHT) / static_cast<float>(frame.rows);
    const float scalingRatio = std::min<float>(std::min<float>(rw, rh), 1.0f);

    cv::Mat downscaled;
    cv::resize(frame, downscaled, cv::Size(), scalingRatio, scalingRatio, cv::INTER_LINEAR);

    cv::Mat input;
    cheshm::normalise_to_u8(downscaled, input);

    const cv::Size workingSize{static_cast<int>(std::floor(scalingRatio * frame.cols)),
                               static_cast<int>(std::floor(scalingRatio * frame.rows))};

    // Pupil-pixel bounds derived from canthi-distance assumptions.
    const float d =
        std::sqrt(static_cast<float>(workingSize.height * workingSize.height + workingSize.width * workingSize.width));
    const float maxCanthiDistancePx = d;
    const float minCanthiDistancePx = 2.0f * d / 3.0f;
    const int maxPupilDiameterPx =
        static_cast<int>(maxCanthiDistancePx * (max_pupil_diameter_mm / canthi_distance_mm));
    const int minPupilDiameterPx =
        static_cast<int>(minCanthiDistancePx * (min_pupil_diameter_mm / canthi_distance_mm));

    // Edge-detection working buffers.
    cv::Mat dx = cv::Mat::zeros(workingSize, CV_32F);
    cv::Mat dy = cv::Mat::zeros(workingSize, CV_32F);
    cv::Mat magnitude = cv::Mat::zeros(workingSize, CV_32F);
    cv::Mat edgeType = cv::Mat::zeros(workingSize, CV_8U);
    cv::Mat edge = cv::Mat::zeros(workingSize, CV_8U);

    cv::Mat detectedEdges = cheshm::canny(input,
                                          dx,
                                          dy,
                                          magnitude,
                                          edgeType,
                                          edge,
                                          defaults::CANNY_BINS,
                                          defaults::CANNY_NON_EDGE_RATIO,
                                          defaults::CANNY_LOW_HIGH_RATIO);
    cheshm::filter_edges(detectedEdges);

    std::vector<PupilCandidate> candidates;
    findPupilEdgeCandidates(input, detectedEdges, candidates, minPupilDiameterPx, maxPupilDiameterPx, outline_bias);
    if (candidates.empty())
        return std::nullopt;

    combineEdgeCandidates(input, candidates, minPupilDiameterPx, maxPupilDiameterPx, outline_bias);

    for (auto& c : candidates)
    {
        if (c.outlineContrast < 0.5f)
            c.score = 0;
        if (c.outline.size.area() > CV_PI * std::pow(0.5f * maxPupilDiameterPx, 2))
            c.score = 0;
        if (c.outline.size.area() < CV_PI * std::pow(0.5f * minPupilDiameterPx, 2))
            c.score = 0;
    }

    std::sort(candidates.begin(), candidates.end());
    PupilCandidate selected = candidates.back();
    if (selected.score == 0)
        return std::nullopt;

    searchInnerCandidates(candidates, selected);

    // Scale back to full-frame coordinates.
    cv::RotatedRect scaledEllipse(
        cv::Point2f(selected.outline.center.x / scalingRatio, selected.outline.center.y / scalingRatio),
        cv::Size2f(selected.outline.size.width / scalingRatio, selected.outline.size.height / scalingRatio),
        selected.outline.angle);

    return DetectResult{scaledEllipse, selected.outlineContrast};
}

} // namespace cheshm::PuRe
