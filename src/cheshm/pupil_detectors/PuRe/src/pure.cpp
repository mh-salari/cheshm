// PuRe pupil detector algorithm body.

#include "PuRe/pure.hpp"
#include "PuRe/defaults.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <utility>
#include <vector>

namespace cheshm::PuRe {
namespace {

using namespace cv;   // NOLINT(google-build-using-namespace)
using namespace std;  // NOLINT(google-build-using-namespace)

// Lookup table for sin / cos at integer-degree resolution. Indexed
// `0..720` so the cos lookup `sinTable[450 - angle]` works for
// `angle in [0, 360]`.
const std::array<float, 721> sinTable = []() {
    std::array<float, 721> t{};
    for (int i = 0; i < 721; ++i) {
        t[i] = std::sin(static_cast<float>((i - 360) * CV_PI / 180.0));
    }
    return t;
}();

inline void sincos_deg(int angle, float &cosval, float &sinval)
{
    angle += (angle < 0 ? 360 : 0);
    sinval = sinTable[360 + angle];
    cosval = sinTable[360 + (90 + angle) % 360];
}

inline std::vector<Point> ellipse_to_points(const RotatedRect &ellipse, int delta)
{
    int angle = static_cast<int>(ellipse.angle);
    while (angle < 0) angle += 360;
    while (angle > 360) angle -= 360;

    float alpha, beta;
    sincos_deg(angle, alpha, beta);

    std::vector<Point> points;
    for (int i = 0; i < 360; i += delta) {
        float cosI, sinI;
        sincos_deg(i, cosI, sinI);
        const float x = 0.5f * ellipse.size.width * cosI;
        const float y = 0.5f * ellipse.size.height * sinI;
        points.emplace_back(
            static_cast<int>(std::roundf(ellipse.center.x + x * alpha - y * beta)),
            static_cast<int>(std::roundf(ellipse.center.y + x * beta + y * alpha)));
    }
    return points;
}

// (1 - cos(22.5°)) / sin(22.5°) — minimum minor/major axis ratio that
// keeps a candidate above the "too flat to be a pupil" gate.
constexpr float MIN_CURVATURE_RATIO = 0.198912f;

struct PupilCandidate {
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

    enum { Q0 = 0, Q1 = 1, Q2 = 2, Q3 = 3 };

    explicit PupilCandidate(std::vector<Point> pts) : points(std::move(pts)) {}

    bool operator<(const PupilCandidate &c) const { return score < c.score; }

    static float ratio(float a, float b)
    {
        std::pair<float, float> sorted = std::minmax(a, b);
        return sorted.first / sorted.second;
    }

    bool isValid(const Mat &intensityImage, int minPupilDiameterPx, int maxPupilDiameterPx, int bias);
    bool fastValidityCheck(int maxPupilDiameterPx);
    bool validityCheck(const Mat &intensityImage, int bias);
    bool validateAnchorDistribution();
    bool validateOutlineContrast(const Mat &intensityImage, int bias);

    void updateScore()
    {
        score = 0.33f * aspectRatio + 0.33f * anchorDistribution + 0.34f * outlineContrast;
    }
};

bool PupilCandidate::isValid(const Mat &intensityImage, int minPupilDiameterPx, int maxPupilDiameterPx, int bias)
{
    if (points.size() < 5) return false;

    float maxGap = 0;
    for (auto p1 = points.begin(); p1 != points.end(); ++p1) {
        for (auto p2 = p1 + 1; p2 != points.end(); ++p2) {
            const float gap = static_cast<float>(norm(*p2 - *p1));
            if (gap > maxGap) maxGap = gap;
        }
    }

    if (maxGap >= maxPupilDiameterPx) return false;
    if (maxGap <= minPupilDiameterPx) return false;

    outline = fitEllipse(points);
    boundaries = {0, 0, intensityImage.cols, intensityImage.rows};

    if (!boundaries.contains(outline.center)) return false;

    if (!fastValidityCheck(maxPupilDiameterPx)) return false;

    pointsMinAreaRect = minAreaRect(points);
    if (ratio(pointsMinAreaRect.size.width, pointsMinAreaRect.size.height) < MIN_CURVATURE_RATIO)
        return false;

    if (!validityCheck(intensityImage, bias)) return false;

    updateScore();
    return true;
}

bool PupilCandidate::fastValidityCheck(int maxPupilDiameterPx)
{
    const std::pair<float, float> axis = std::minmax(outline.size.width, outline.size.height);
    minorAxis = axis.first;
    majorAxis = axis.second;
    aspectRatio = minorAxis / majorAxis;

    if (aspectRatio < MIN_CURVATURE_RATIO) return false;
    if (majorAxis > maxPupilDiameterPx) return false;

    combinationRegion = boundingRect(points);
    combinationRegion.width = std::max<int>(combinationRegion.width, combinationRegion.height);
    combinationRegion.height = combinationRegion.width;

    return true;
}

bool PupilCandidate::validateOutlineContrast(const Mat &intensityImage, int bias)
{
    const int delta = static_cast<int>(0.15f * minorAxis);
    const Point c = outline.center;

    int evaluated = 0;
    int validCount = 0;

    const std::vector<Point> outlinePoints = ellipse_to_points(outline, 10);
    for (const Point &p : outlinePoints) {
        const int dxp = p.x - c.x;
        const int dyp = p.y - c.y;

        float a = 0.0f;
        if (dxp != 0) a = static_cast<float>(dyp) / static_cast<float>(dxp);
        const float b = c.y - a * c.x;

        if (a == 0.0f) continue;

        if (std::abs(dxp) > std::abs(dyp)) {
            const int sx = p.x - delta;
            const int ex = p.x + delta;
            const int sy = static_cast<int>(std::roundf(a * sx + b));
            const int ey = static_cast<int>(std::roundf(a * ex + b));
            const Point start{sx, sy};
            const Point end{ex, ey};
            ++evaluated;

            if (!boundaries.contains(start) || !boundaries.contains(end)) continue;

            float m1 = 0.0f;
            for (int x = sx; x < p.x; ++x)
                m1 += intensityImage.ptr<uchar>(static_cast<int>(std::roundf(a * x + b)))[x];
            m1 = std::roundf(m1 / delta);

            float m2 = 0.0f;
            for (int x = p.x + 1; x <= ex; ++x)
                m2 += intensityImage.ptr<uchar>(static_cast<int>(std::roundf(a * x + b)))[x];
            m2 = std::roundf(m2 / delta);

            if (p.x < c.x) {
                if (m1 > m2 + bias) ++validCount;
            } else {
                if (m2 > m1 + bias) ++validCount;
            }
        } else {
            const int sy = p.y - delta;
            const int ey = p.y + delta;
            const int sx = static_cast<int>(std::roundf((sy - b) / a));
            const int ex = static_cast<int>(std::roundf((ey - b) / a));
            const Point start{sx, sy};
            const Point end{ex, ey};
            ++evaluated;

            if (!boundaries.contains(start) || !boundaries.contains(end)) continue;

            float m1 = 0.0f;
            for (int y = sy; y < p.y; ++y)
                m1 += intensityImage.ptr<uchar>(y)[static_cast<int>(std::roundf((y - b) / a))];
            m1 = std::roundf(m1 / delta);

            float m2 = 0.0f;
            for (int y = p.y + 1; y <= ey; ++y)
                m2 += intensityImage.ptr<uchar>(y)[static_cast<int>(std::roundf((y - b) / a))];
            m2 = std::roundf(m2 / delta);

            if (p.y < c.y) {
                if (m1 > m2 + bias) ++validCount;
            } else {
                if (m2 > m1 + bias) ++validCount;
            }
        }
    }
    if (evaluated == 0) return false;
    outlineContrast = static_cast<float>(validCount) / static_cast<float>(evaluated);
    return true;
}

bool PupilCandidate::validateAnchorDistribution()
{
    anchorPointSlices.reset();
    for (const Point &p : points) {
        if (p.x - outline.center.x < 0) {
            if (p.y - outline.center.y < 0) anchorPointSlices.set(Q0);
            else anchorPointSlices.set(Q3);
        } else {
            if (p.y - outline.center.y < 0) anchorPointSlices.set(Q1);
            else anchorPointSlices.set(Q2);
        }
    }
    anchorDistribution = static_cast<float>(anchorPointSlices.count()) / static_cast<float>(anchorPointSlices.size());
    return true;
}

bool PupilCandidate::validityCheck(const Mat &intensityImage, int bias)
{
    mp = std::accumulate(points.begin(), points.end(), Point(0, 0));
    mp.x = std::roundf(mp.x / points.size());
    mp.y = std::roundf(mp.y / points.size());

    outline.points(v);
    const std::vector<cv::Point2f> pv(v, v + 4);
    if (cv::pointPolygonTest(pv, mp, false) <= 0) return false;

    if (!validateAnchorDistribution()) return false;

    if (!validateOutlineContrast(intensityImage, bias)) return false;

    return true;
}

Mat canny(
    const Mat &in,
    Mat &dx, Mat &dy, Mat &magnitude, Mat &edgeType, Mat &edge,
    int bins, float nonEdgePixelsRatio, float lowHighThresholdRatio)
{
    Mat blurred;
    GaussianBlur(in, blurred, Size(5, 5), 1.5, 1.5, BORDER_REPLICATE);

    Sobel(blurred, dx, dx.type(), 1, 0, 7, 1, BORDER_REPLICATE);
    Sobel(blurred, dy, dy.type(), 0, 1, 7, 1, BORDER_REPLICATE);

    double minMag = 0;
    double maxMag = 0;
    cv::magnitude(dx, dy, magnitude);
    cv::minMaxLoc(magnitude, &minMag, &maxMag);

    magnitude = magnitude / maxMag;

    std::vector<int> histogram(bins, 0);
    Mat res_idx = (bins - 1) * magnitude;
    res_idx.convertTo(res_idx, CV_16U);
    for (int i = 0; i < res_idx.rows; ++i) {
        const short *p = res_idx.ptr<short>(i);
        for (int j = 0; j < res_idx.cols; ++j)
            ++histogram[p[j]];
    }

    int sum = 0;
    const int nonEdgePixels = static_cast<int>(nonEdgePixelsRatio * in.rows * in.cols);
    float high_th = 0;
    for (int i = 0; i < bins; ++i) {
        sum += histogram[i];
        if (sum > nonEdgePixels) {
            high_th = static_cast<float>(i + 1) / bins;
            break;
        }
    }
    const float low_th = lowHighThresholdRatio * high_th;

    // Non-maximum suppression.
    const float tg22_5 = 0.4142135623730950488016887242097f;
    const float tg67_5 = 2.4142135623730950488016887242097f;
    edgeType.setTo(0);
    for (int i = 1; i < magnitude.rows - 1; ++i) {
        uchar *_edgeType = edgeType.ptr<uchar>(i);
        const float *p_res = magnitude.ptr<float>(i);
        const float *p_res_t = magnitude.ptr<float>(i - 1);
        const float *p_res_b = magnitude.ptr<float>(i + 1);
        const float *p_x = dx.ptr<float>(i);
        const float *p_y = dy.ptr<float>(i);

        for (int j = 1; j < magnitude.cols - 1; ++j) {
            const float m = p_res[j];
            if (m < low_th) continue;

            const float iy = p_y[j];
            const float ix = p_x[j];
            const float y = std::abs(iy);
            const float x = std::abs(ix);

            const uchar val = p_res[j] > high_th ? 255 : 128;

            const float tg22_5x = tg22_5 * x;
            if (y < tg22_5x) {
                if (m > p_res[j - 1] && m >= p_res[j + 1]) _edgeType[j] = val;
            } else {
                const float tg67_5x = tg67_5 * x;
                if (y > tg67_5x) {
                    if (m > p_res_b[j] && m >= p_res_t[j]) _edgeType[j] = val;
                } else {
                    if ((iy <= 0) == (ix <= 0)) {
                        if (m > p_res_t[j - 1] && m >= p_res_b[j + 1]) _edgeType[j] = val;
                    } else {
                        if (m > p_res_b[j - 1] && m >= p_res_t[j + 1]) _edgeType[j] = val;
                    }
                }
            }
        }
    }

    // Hysteresis.
    const int pic_x = edgeType.cols;
    const int pic_y = edgeType.rows;
    const int area = pic_x * pic_y;
    int lines_idx = 0;
    int idx = 0;

    std::vector<int> lines;
    edge.setTo(0);
    for (int i = 1; i < pic_y - 1; ++i) {
        for (int j = 1; j < pic_x - 1; ++j) {
            if (edgeType.data[idx + j] != 255 || edge.data[idx + j] != 0)
                continue;

            edge.data[idx + j] = 255;
            lines_idx = 1;
            lines.clear();
            lines.push_back(idx + j);
            int akt_idx = 0;

            while (akt_idx < lines_idx) {
                const int akt_pos = lines[akt_idx];
                ++akt_idx;

                if (akt_pos - pic_x - 1 < 0 || akt_pos + pic_x + 1 >= area) continue;

                for (int k1 = -1; k1 < 2; ++k1)
                    for (int k2 = -1; k2 < 2; ++k2) {
                        const int neighbour = (akt_pos + (k1 * pic_x)) + k2;
                        if (edge.data[neighbour] != 0 || edgeType.data[neighbour] == 0) continue;
                        edge.data[neighbour] = 255;
                        lines.push_back(neighbour);
                        ++lines_idx;
                    }
            }
        }
        idx += pic_x;
    }

    return edge;
}

// Cleans the edge image in three passes:
//   1) Thin 2x2 corner clusters down to a single edge pixel.
//   2) Drop pixels with more than 3 lit neighbours.
//   3) Local-pattern rewrites that straighten short staircase / spur
//      segments produced by the previous Canny step.
void filterEdges(Mat &edges)
{
    const int start_x = 5;
    const int start_y = 5;
    const int end_x = edges.cols - 5;
    const int end_y = edges.rows - 5;

    for (int j = start_y; j < end_y; ++j)
        for (int i = start_x; i < end_x; ++i) {
            uchar box[9];
            box[4] = edges.data[(edges.cols * j) + i];
            if (box[4]) {
                box[1] = edges.data[(edges.cols * (j - 1)) + i];
                box[3] = edges.data[(edges.cols * j) + (i - 1)];
                box[5] = edges.data[(edges.cols * j) + (i + 1)];
                box[7] = edges.data[(edges.cols * (j + 1)) + i];

                if (box[5] && box[7]) edges.data[(edges.cols * j) + i] = 0;
                if (box[5] && box[1]) edges.data[(edges.cols * j) + i] = 0;
                if (box[3] && box[7]) edges.data[(edges.cols * j) + i] = 0;
                if (box[3] && box[1]) edges.data[(edges.cols * j) + i] = 0;
            }
        }

    for (int j = start_y; j < end_y; ++j)
        for (int i = start_x; i < end_x; ++i) {
            int neig = 0;
            for (int k1 = -1; k1 < 2; ++k1)
                for (int k2 = -1; k2 < 2; ++k2)
                    if (edges.data[(edges.cols * (j + k1)) + (i + k2)] > 0) ++neig;
            if (neig > 3) edges.data[(edges.cols * j) + i] = 0;
        }

    for (int j = start_y; j < end_y; ++j)
        for (int i = start_x; i < end_x; ++i) {
            uchar box[17];
            box[4] = edges.data[(edges.cols * j) + i];
            if (box[4]) {
                box[0] = edges.data[(edges.cols * (j - 1)) + (i - 1)];
                box[1] = edges.data[(edges.cols * (j - 1)) + i];
                box[2] = edges.data[(edges.cols * (j - 1)) + (i + 1)];
                box[3] = edges.data[(edges.cols * j) + (i - 1)];
                box[5] = edges.data[(edges.cols * j) + (i + 1)];
                box[6] = edges.data[(edges.cols * (j + 1)) + (i - 1)];
                box[7] = edges.data[(edges.cols * (j + 1)) + i];
                box[8] = edges.data[(edges.cols * (j + 1)) + (i + 1)];

                box[9] = edges.data[(edges.cols * j) + (i + 2)];
                box[10] = edges.data[(edges.cols * (j + 2)) + i];
                box[11] = edges.data[(edges.cols * j) + (i + 3)];
                box[12] = edges.data[(edges.cols * (j - 1)) + (i + 2)];
                box[13] = edges.data[(edges.cols * (j + 1)) + (i + 2)];
                box[14] = edges.data[(edges.cols * (j + 3)) + i];
                box[15] = edges.data[(edges.cols * (j + 2)) + (i - 1)];
                box[16] = edges.data[(edges.cols * (j + 2)) + (i + 1)];

                if ((box[10] && !box[7]) && (box[8] || box[6])) {
                    edges.data[(edges.cols * (j + 1)) + (i - 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + i] = 255;
                }
                if ((box[14] && !box[7] && !box[10]) && ((box[8] || box[6]) && (box[16] || box[15]))) {
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + (i - 1)] = 0;
                    edges.data[(edges.cols * (j + 2)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 2)) + (i - 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + i] = 255;
                    edges.data[(edges.cols * (j + 2)) + i] = 255;
                }
                if ((box[9] && !box[5]) && (box[8] || box[2])) {
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j - 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * j) + (i + 1)] = 255;
                }
                if ((box[11] && !box[5] && !box[9]) && ((box[8] || box[2]) && (box[13] || box[12]))) {
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j - 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + (i + 2)] = 0;
                    edges.data[(edges.cols * (j - 1)) + (i + 2)] = 0;
                    edges.data[(edges.cols * j) + (i + 1)] = 255;
                    edges.data[(edges.cols * j) + (i + 2)] = 255;
                }
            }
        }

    for (int j = start_y; j < end_y; ++j)
        for (int i = start_x; i < end_x; ++i) {
            uchar box[33];
            box[4] = edges.data[(edges.cols * j) + i];
            if (box[4]) {
                box[0] = edges.data[(edges.cols * (j - 1)) + (i - 1)];
                box[1] = edges.data[(edges.cols * (j - 1)) + i];
                box[2] = edges.data[(edges.cols * (j - 1)) + (i + 1)];
                box[3] = edges.data[(edges.cols * j) + (i - 1)];
                box[5] = edges.data[(edges.cols * j) + (i + 1)];
                box[6] = edges.data[(edges.cols * (j + 1)) + (i - 1)];
                box[7] = edges.data[(edges.cols * (j + 1)) + i];
                box[8] = edges.data[(edges.cols * (j + 1)) + (i + 1)];

                box[9] = edges.data[(edges.cols * (j - 1)) + (i + 2)];
                box[10] = edges.data[(edges.cols * (j - 1)) + (i - 2)];
                box[11] = edges.data[(edges.cols * (j + 1)) + (i + 2)];
                box[12] = edges.data[(edges.cols * (j + 1)) + (i - 2)];

                box[13] = edges.data[(edges.cols * (j - 2)) + (i - 1)];
                box[14] = edges.data[(edges.cols * (j - 2)) + (i + 1)];
                box[15] = edges.data[(edges.cols * (j + 2)) + (i - 1)];
                box[16] = edges.data[(edges.cols * (j + 2)) + (i + 1)];

                box[17] = edges.data[(edges.cols * (j - 3)) + (i - 1)];
                box[18] = edges.data[(edges.cols * (j - 3)) + (i + 1)];
                box[19] = edges.data[(edges.cols * (j + 3)) + (i - 1)];
                box[20] = edges.data[(edges.cols * (j + 3)) + (i + 1)];

                box[21] = edges.data[(edges.cols * (j + 1)) + (i + 3)];
                box[22] = edges.data[(edges.cols * (j + 1)) + (i - 3)];
                box[23] = edges.data[(edges.cols * (j - 1)) + (i + 3)];
                box[24] = edges.data[(edges.cols * (j - 1)) + (i - 3)];

                box[25] = edges.data[(edges.cols * (j - 2)) + (i - 2)];
                box[26] = edges.data[(edges.cols * (j + 2)) + (i + 2)];
                box[27] = edges.data[(edges.cols * (j - 2)) + (i + 2)];
                box[28] = edges.data[(edges.cols * (j + 2)) + (i - 2)];

                if (box[7] && box[2] && box[9]) edges.data[(edges.cols * j) + i] = 0;
                if (box[7] && box[0] && box[10]) edges.data[(edges.cols * j) + i] = 0;
                if (box[1] && box[8] && box[11]) edges.data[(edges.cols * j) + i] = 0;
                if (box[1] && box[6] && box[12]) edges.data[(edges.cols * j) + i] = 0;

                if (box[0] && box[13] && box[17] && box[8] && box[11] && box[21]) edges.data[(edges.cols * j) + i] = 0;
                if (box[2] && box[14] && box[18] && box[6] && box[12] && box[22]) edges.data[(edges.cols * j) + i] = 0;
                if (box[6] && box[15] && box[19] && box[2] && box[9] && box[23]) edges.data[(edges.cols * j) + i] = 0;
                if (box[8] && box[16] && box[20] && box[0] && box[10] && box[24]) edges.data[(edges.cols * j) + i] = 0;

                if (box[0] && box[25] && box[2] && box[27]) edges.data[(edges.cols * j) + i] = 0;
                if (box[0] && box[25] && box[6] && box[28]) edges.data[(edges.cols * j) + i] = 0;
                if (box[8] && box[26] && box[2] && box[27]) edges.data[(edges.cols * j) + i] = 0;
                if (box[8] && box[26] && box[6] && box[28]) edges.data[(edges.cols * j) + i] = 0;

                uchar box2[18];
                box2[1] = edges.data[(edges.cols * j) + (i - 1)];
                box2[2] = edges.data[(edges.cols * (j - 1)) + (i - 2)];
                box2[3] = edges.data[(edges.cols * (j - 2)) + (i - 3)];
                box2[4] = edges.data[(edges.cols * (j - 1)) + (i + 1)];
                box2[5] = edges.data[(edges.cols * (j - 2)) + (i + 2)];
                box2[6] = edges.data[(edges.cols * (j + 1)) + (i - 2)];
                box2[7] = edges.data[(edges.cols * (j + 2)) + (i - 3)];
                box2[8] = edges.data[(edges.cols * (j + 1)) + (i + 1)];
                box2[9] = edges.data[(edges.cols * (j + 2)) + (i + 2)];
                box2[10] = edges.data[(edges.cols * (j + 1)) + i];
                box2[15] = edges.data[(edges.cols * (j - 1)) + (i - 1)];
                box2[16] = edges.data[(edges.cols * (j - 2)) + (i - 2)];
                box2[11] = edges.data[(edges.cols * (j + 2)) + (i + 1)];
                box2[12] = edges.data[(edges.cols * (j + 3)) + (i + 2)];
                box2[13] = edges.data[(edges.cols * (j + 2)) + (i - 1)];
                box2[14] = edges.data[(edges.cols * (j + 3)) + (i - 2)];

                if (box2[1] && box2[2] && box2[3] && box2[4] && box2[5]) edges.data[(edges.cols * j) + i] = 0;
                if (box2[1] && box2[6] && box2[7] && box2[8] && box2[9]) edges.data[(edges.cols * j) + i] = 0;
                if (box2[10] && box2[11] && box2[12] && box2[4] && box2[5]) edges.data[(edges.cols * j) + i] = 0;
                if (box2[10] && box2[13] && box2[14] && box2[15] && box2[16]) edges.data[(edges.cols * j) + i] = 0;
            }
        }
}

inline int pointHash(Point p, int cols) { return p.y * cols + p.x; }

void removeDuplicates(std::vector<std::vector<Point>> &curves, int cols)
{
    std::map<int, uchar> contourMap;
    for (std::size_t i = curves.size(); i-- > 0;) {
        if (contourMap.count(pointHash(curves[i][0], cols)) > 0) {
            curves.erase(curves.begin() + i);
        } else {
            for (std::size_t j = 0; j < curves[i].size(); ++j)
                contourMap[pointHash(curves[i][j], cols)] = 1;
        }
    }
}

void findPupilEdgeCandidates(
    const Mat &intensityImage, Mat &edge,
    std::vector<PupilCandidate> &candidates,
    int minPupilDiameterPx, int maxPupilDiameterPx, int outlineBias)
{
    std::vector<Vec4i> hierarchy;
    std::vector<std::vector<Point>> curves;
    findContours(edge, curves, hierarchy, RETR_LIST, CHAIN_APPROX_TC89_KCOS);

    removeDuplicates(curves, edge.cols);

    for (std::size_t i = curves.size(); i-- > 0;) {
        PupilCandidate candidate(curves[i]);
        if (candidate.isValid(intensityImage, minPupilDiameterPx, maxPupilDiameterPx, outlineBias))
            candidates.push_back(candidate);
    }
}

void combineEdgeCandidates(
    const Mat &intensityImage,
    std::vector<PupilCandidate> &candidates,
    int minPupilDiameterPx, int maxPupilDiameterPx, int outlineBias)
{
    if (candidates.size() <= 1) return;

    std::vector<PupilCandidate> mergedCandidates;
    for (auto pc = candidates.begin(); pc != candidates.end(); ++pc) {
        for (auto pc2 = pc + 1; pc2 != candidates.end(); ++pc2) {
            const Rect intersection = pc->combinationRegion & pc2->combinationRegion;
            if (intersection.area() < 1) continue;

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

void searchInnerCandidates(const std::vector<PupilCandidate> &candidates, PupilCandidate &selected)
{
    if (candidates.size() <= 1) return;

    const float searchRadius = 0.5f * selected.majorAxis;
    std::vector<PupilCandidate> insiders;
    for (const auto &pc : candidates) {
        if (searchRadius < pc.majorAxis) continue;
        if (norm(selected.outline.center - pc.outline.center) > searchRadius) continue;
        if (pc.outlineContrast < 0.75f) continue;
        insiders.push_back(pc);
    }
    if (insiders.empty()) return;

    std::sort(insiders.begin(), insiders.end());
    selected = insiders.back();
}

}  // namespace

std::optional<DetectResult> detect(
    const cv::Mat &frame,
    float min_pupil_diameter_mm,
    float max_pupil_diameter_mm,
    float canthi_distance_mm,
    int outline_bias)
{
    if (frame.empty()) return std::nullopt;

    // Downscale to working size.
    const float rw = static_cast<float>(defaults::BASE_WIDTH) / static_cast<float>(frame.cols);
    const float rh = static_cast<float>(defaults::BASE_HEIGHT) / static_cast<float>(frame.rows);
    const float scalingRatio = std::min<float>(std::min<float>(rw, rh), 1.0f);

    cv::Mat downscaled;
    cv::resize(frame, downscaled, cv::Size(), scalingRatio, scalingRatio, cv::INTER_LINEAR);

    cv::Mat input;
    cv::normalize(downscaled, input, 0, 255, cv::NORM_MINMAX, CV_8U);

    const cv::Size workingSize{
        static_cast<int>(std::floor(scalingRatio * frame.cols)),
        static_cast<int>(std::floor(scalingRatio * frame.rows))};

    // Pupil-pixel bounds derived from canthi-distance assumptions.
    const float d = std::sqrt(static_cast<float>(workingSize.height * workingSize.height +
                                                  workingSize.width * workingSize.width));
    const float maxCanthiDistancePx = d;
    const float minCanthiDistancePx = 2.0f * d / 3.0f;
    const int maxPupilDiameterPx = static_cast<int>(maxCanthiDistancePx * (max_pupil_diameter_mm / canthi_distance_mm));
    const int minPupilDiameterPx = static_cast<int>(minCanthiDistancePx * (min_pupil_diameter_mm / canthi_distance_mm));

    // Edge-detection working buffers.
    cv::Mat dx = cv::Mat::zeros(workingSize, CV_32F);
    cv::Mat dy = cv::Mat::zeros(workingSize, CV_32F);
    cv::Mat magnitude = cv::Mat::zeros(workingSize, CV_32F);
    cv::Mat edgeType = cv::Mat::zeros(workingSize, CV_8U);
    cv::Mat edge = cv::Mat::zeros(workingSize, CV_8U);

    cv::Mat detectedEdges = canny(input, dx, dy, magnitude, edgeType, edge,
                                  defaults::CANNY_BINS,
                                  defaults::CANNY_NON_EDGE_RATIO,
                                  defaults::CANNY_LOW_HIGH_RATIO);
    filterEdges(detectedEdges);

    std::vector<PupilCandidate> candidates;
    findPupilEdgeCandidates(input, detectedEdges, candidates,
                            minPupilDiameterPx, maxPupilDiameterPx, outline_bias);
    if (candidates.empty()) return std::nullopt;

    combineEdgeCandidates(input, candidates,
                          minPupilDiameterPx, maxPupilDiameterPx, outline_bias);

    for (auto &c : candidates) {
        if (c.outlineContrast < 0.5f) c.score = 0;
        if (c.outline.size.area() > CV_PI * std::pow(0.5f * maxPupilDiameterPx, 2)) c.score = 0;
        if (c.outline.size.area() < CV_PI * std::pow(0.5f * minPupilDiameterPx, 2)) c.score = 0;
    }

    std::sort(candidates.begin(), candidates.end());
    PupilCandidate selected = candidates.back();
    if (selected.score == 0) return std::nullopt;

    searchInnerCandidates(candidates, selected);

    // Scale back to full-frame coordinates.
    cv::RotatedRect scaledEllipse(
        cv::Point2f(selected.outline.center.x / scalingRatio,
                    selected.outline.center.y / scalingRatio),
        cv::Size2f(selected.outline.size.width / scalingRatio,
                   selected.outline.size.height / scalingRatio),
        selected.outline.angle);

    return DetectResult{scaledEllipse, selected.outlineContrast};
}

}  // namespace cheshm::PuRe