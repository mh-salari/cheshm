// ExCuSe pupil detector algorithm body.

#include "ExCuSe/excuse.hpp"

#include "cheshm/canny_gaussian16.hpp"
#include "cheshm/ellipse_intensity_gap.hpp"

#include "ExCuSe/defaults.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>

namespace cheshm::ExCuSe
{
namespace
{

using namespace cv;  // NOLINT(google-build-using-namespace)
using namespace std; // NOLINT(google-build-using-namespace)

using defaults::DEF_SIZE;
using defaults::IMG_SIZE;


bool peek(const cv::Mat& pic,
          double& stddev,
          int start_x,
          int end_x,
          int start_y,
          int end_y,
          int peek_detector_factor,
          int bright_region_th)
{
    int gray_hist[256];
    int max_gray = 0;
    int max_gray_pos = 0;
    int mean_gray = 0;
    int mean_gray_cnt = 0;

    for (int i = 0; i < 256; i++)
        gray_hist[i] = 0;

    double mean_feld[1000];
    double std_feld[1000];
    for (int i = start_x; i < end_x; i++)
    {
        mean_feld[i] = 0;
        std_feld[i] = 0;
    }

    for (int i = start_x; i < end_x; i++)
        for (int j = start_y; j < end_y; j++)
        {
            int idx = static_cast<int>(pic.data[(pic.cols * j) + i]);
            gray_hist[idx]++;
            mean_feld[i] += idx;
        }

    for (int i = start_x; i < end_x; i++)
        mean_feld[i] = (mean_feld[i] / double(end_y - start_y));

    for (int i = start_x; i < end_x; i++)
        for (int j = start_y; j < end_y; j++)
        {
            int idx = static_cast<int>(pic.data[(pic.cols * j) + i]);
            std_feld[i] += (mean_feld[i] - idx) * (mean_feld[i] - idx);
        }

    for (int i = start_x; i < end_x; i++)
        std_feld[i] = sqrt(std_feld[i] / double(end_y - start_y));

    stddev = 0;
    for (int i = start_x; i < end_x; i++)
    {
        stddev += std_feld[i];
    }

    stddev = stddev / ((end_x - start_x));

    for (int i = 0; i < 256; i++)
        if (gray_hist[i] > 0)
        {
            mean_gray += gray_hist[i];
            mean_gray_cnt++;

            if (max_gray < gray_hist[i])
            {
                max_gray = gray_hist[i];
                max_gray_pos = i;
            }
        }

    if (mean_gray_cnt < 1)
        mean_gray_cnt = 1;

    mean_gray = ceil(static_cast<double>(mean_gray) / static_cast<double>(mean_gray_cnt));

    if (max_gray > (mean_gray * peek_detector_factor) && max_gray_pos > bright_region_th)
        return true;
    else
        return false;
}

void remove_points_with_low_angle(cv::Mat& edge, int start_xx, int end_xx, int start_yy, int end_yy)
{
    int start_x = start_xx + 5;
    int end_x = end_xx - 5;
    int start_y = start_yy + 5;
    int end_y = end_yy - 5;

    if (start_x < 5)
        start_x = 5;
    if (end_x > edge.cols - 5)
        end_x = edge.cols - 5;
    if (start_y < 5)
        start_y = 5;
    if (end_y > edge.rows - 5)
        end_y = edge.rows - 5;

    for (int j = start_y; j < end_y; j++)
        for (int i = start_x; i < end_x; i++)
        {
            if (static_cast<int>(edge.data[(edge.cols * (j)) + (i)]))
            {
                int box[8];

                box[0] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i - 1)]);
                box[1] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i)]);
                box[2] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i + 1)]);
                box[3] = static_cast<int>(edge.data[(edge.cols * (j)) + (i + 1)]);
                box[4] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i + 1)]);
                box[5] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i)]);
                box[6] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i - 1)]);
                box[7] = static_cast<int>(edge.data[(edge.cols * (j)) + (i - 1)]);

                bool valid = false;

                for (int k = 0; k < 8 && !valid; k++)
                    if (box[k] && (box[(k + 2) % 8] || box[(k + 3) % 8] || box[(k + 4) % 8] || box[(k + 5) % 8] ||
                                   box[(k + 6) % 8]))
                        valid = true;

                if (!valid)
                    edge.data[(edge.cols * (j)) + (i)] = 0;
            }
        }

    for (int j = start_y; j < end_y; j++)
        for (int i = start_x; i < end_x; i++)
        {
            int box[9];

            box[4] = static_cast<int>(edge.data[(edge.cols * (j)) + (i)]);

            if (box[4])
            {
                box[1] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i)]);
                box[3] = static_cast<int>(edge.data[(edge.cols * (j)) + (i - 1)]);
                box[5] = static_cast<int>(edge.data[(edge.cols * (j)) + (i + 1)]);
                box[7] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i)]);

                if ((box[5] && box[7]))
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if ((box[5] && box[1]))
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if ((box[3] && box[7]))
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if ((box[3] && box[1]))
                    edge.data[(edge.cols * (j)) + (i)] = 0;

                //		edge.data[(edge.cols*(j))+(i)]=0;
            }
        }

    for (int j = start_y; j < end_y; j++)
        for (int i = start_x; i < end_x; i++)
        {
            int box[17];

            box[4] = static_cast<int>(edge.data[(edge.cols * (j)) + (i)]);

            if (box[4])
            {
                box[0] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i - 1)]);
                box[1] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i)]);
                box[2] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i + 1)]);

                box[3] = static_cast<int>(edge.data[(edge.cols * (j)) + (i - 1)]);
                box[5] = static_cast<int>(edge.data[(edge.cols * (j)) + (i + 1)]);

                box[6] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i - 1)]);
                box[7] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i)]);
                box[8] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i + 1)]);

                // external
                box[9] = static_cast<int>(edge.data[(edge.cols * (j)) + (i + 2)]);
                box[10] = static_cast<int>(edge.data[(edge.cols * (j + 2)) + (i)]);

                box[11] = static_cast<int>(edge.data[(edge.cols * (j)) + (i + 3)]);
                box[12] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i + 2)]);
                box[13] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i + 2)]);

                box[14] = static_cast<int>(edge.data[(edge.cols * (j + 3)) + (i)]);
                box[15] = static_cast<int>(edge.data[(edge.cols * (j + 2)) + (i - 1)]);
                box[16] = static_cast<int>(edge.data[(edge.cols * (j + 2)) + (i + 1)]);

                if ((box[10] && !box[7]) && (box[8] || box[6]))
                {
                    edge.data[(edge.cols * (j + 1)) + (i - 1)] = 0;
                    edge.data[(edge.cols * (j + 1)) + (i + 1)] = 0;
                    edge.data[(edge.cols * (j + 1)) + (i)] = 255;
                }

                if ((box[14] && !box[7] && !box[10]) && ((box[8] || box[6]) && (box[16] || box[15])))
                {
                    edge.data[(edge.cols * (j + 1)) + (i + 1)] = 0;
                    edge.data[(edge.cols * (j + 1)) + (i - 1)] = 0;
                    edge.data[(edge.cols * (j + 2)) + (i + 1)] = 0;
                    edge.data[(edge.cols * (j + 2)) + (i - 1)] = 0;
                    edge.data[(edge.cols * (j + 1)) + (i)] = 255;
                    edge.data[(edge.cols * (j + 2)) + (i)] = 255;
                }

                if ((box[9] && !box[5]) && (box[8] || box[2]))
                {
                    edge.data[(edge.cols * (j + 1)) + (i + 1)] = 0;
                    edge.data[(edge.cols * (j - 1)) + (i + 1)] = 0;
                    edge.data[(edge.cols * (j)) + (i + 1)] = 255;
                }

                if ((box[11] && !box[5] && !box[9]) && ((box[8] || box[2]) && (box[13] || box[12])))
                {
                    edge.data[(edge.cols * (j + 1)) + (i + 1)] = 0;
                    edge.data[(edge.cols * (j - 1)) + (i + 1)] = 0;
                    edge.data[(edge.cols * (j + 1)) + (i + 2)] = 0;
                    edge.data[(edge.cols * (j - 1)) + (i + 2)] = 0;
                    edge.data[(edge.cols * (j)) + (i + 1)] = 255;
                    edge.data[(edge.cols * (j)) + (i + 2)] = 255;
                }
            }
        }

    for (int j = start_y; j < end_y; j++)
        for (int i = start_x; i < end_x; i++)
        {
            int box[33];

            box[4] = static_cast<int>(edge.data[(edge.cols * (j)) + (i)]);

            if (box[4])
            {
                box[0] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i - 1)]);
                box[1] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i)]);
                box[2] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i + 1)]);

                box[3] = static_cast<int>(edge.data[(edge.cols * (j)) + (i - 1)]);
                box[5] = static_cast<int>(edge.data[(edge.cols * (j)) + (i + 1)]);

                box[6] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i - 1)]);
                box[7] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i)]);
                box[8] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i + 1)]);

                box[9] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i + 2)]);
                box[10] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i - 2)]);
                box[11] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i + 2)]);
                box[12] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i - 2)]);

                box[13] = static_cast<int>(edge.data[(edge.cols * (j - 2)) + (i - 1)]);
                box[14] = static_cast<int>(edge.data[(edge.cols * (j - 2)) + (i + 1)]);
                box[15] = static_cast<int>(edge.data[(edge.cols * (j + 2)) + (i - 1)]);
                box[16] = static_cast<int>(edge.data[(edge.cols * (j + 2)) + (i + 1)]);

                box[17] = static_cast<int>(edge.data[(edge.cols * (j - 3)) + (i - 1)]);
                box[18] = static_cast<int>(edge.data[(edge.cols * (j - 3)) + (i + 1)]);
                box[19] = static_cast<int>(edge.data[(edge.cols * (j + 3)) + (i - 1)]);
                box[20] = static_cast<int>(edge.data[(edge.cols * (j + 3)) + (i + 1)]);

                box[21] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i + 3)]);
                box[22] = static_cast<int>(edge.data[(edge.cols * (j + 1)) + (i - 3)]);
                box[23] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i + 3)]);
                box[24] = static_cast<int>(edge.data[(edge.cols * (j - 1)) + (i - 3)]);

                box[25] = static_cast<int>(edge.data[(edge.cols * (j - 2)) + (i - 2)]);
                box[26] = static_cast<int>(edge.data[(edge.cols * (j + 2)) + (i + 2)]);
                box[27] = static_cast<int>(edge.data[(edge.cols * (j - 2)) + (i + 2)]);
                box[28] = static_cast<int>(edge.data[(edge.cols * (j + 2)) + (i - 2)]);

                box[29] = static_cast<int>(edge.data[(edge.cols * (j - 3)) + (i - 3)]);
                box[30] = static_cast<int>(edge.data[(edge.cols * (j + 3)) + (i + 3)]);
                box[31] = static_cast<int>(edge.data[(edge.cols * (j - 3)) + (i + 3)]);
                box[32] = static_cast<int>(edge.data[(edge.cols * (j + 3)) + (i - 3)]);

                if (box[7] && box[2] && box[9])
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if (box[7] && box[0] && box[10])
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if (box[1] && box[8] && box[11])
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if (box[1] && box[6] && box[12])
                    edge.data[(edge.cols * (j)) + (i)] = 0;

                if (box[0] && box[13] && box[17] && box[8] && box[11] && box[21])
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if (box[2] && box[14] && box[18] && box[6] && box[12] && box[22])
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if (box[6] && box[15] && box[19] && box[2] && box[9] && box[23])
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if (box[8] && box[16] && box[20] && box[0] && box[10] && box[24])
                    edge.data[(edge.cols * (j)) + (i)] = 0;

                if (box[0] && box[25] && box[29] && box[2] && box[27] && box[31])
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if (box[0] && box[25] && box[29] && box[6] && box[28] && box[32])
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if (box[8] && box[26] && box[30] && box[2] && box[27] && box[31])
                    edge.data[(edge.cols * (j)) + (i)] = 0;
                if (box[8] && box[26] && box[30] && box[6] && box[28] && box[32])
                    edge.data[(edge.cols * (j)) + (i)] = 0;
            }
        }
}

std::vector<std::vector<cv::Point>> get_curves(const cv::Mat& pic,
                                               const cv::Mat& edge,
                                               int start_x,
                                               int end_x,
                                               int start_y,
                                               int end_y,
                                               double mean_dist,
                                               int inner_color_range)
{
    std::vector<std::vector<cv::Point>> all_curves;
    std::vector<cv::Point> curve;

    if (start_x < 2)
        start_x = 2;
    if (start_y < 2)
        start_y = 2;
    if (end_x > pic.cols - 2)
        end_x = pic.cols - 2;
    if (end_y > pic.rows - 2)
        end_y = pic.rows - 2;

    int curve_idx = 0;
    cv::Point mean_p;
    bool add_curve;
    int mean_inner_gray;
    int mean_inner_gray_last = 1000000;

    // curve.reserve(1000);
    // all_curves.reserve(1000);

    all_curves.clear();

    bool check[IMG_SIZE][IMG_SIZE];

    for (int i = 0; i < IMG_SIZE; i++)
        for (int j = 0; j < IMG_SIZE; j++)
            check[i][j] = 0;

    for (int i = start_x; i < end_x; i++)
        for (int j = start_y; j < end_y; j++)
        {
            if (edge.data[(edge.cols * (j)) + (i)] == 255 && !check[i][j])
            {
                check[i][j] = 1;

                curve.clear();
                curve_idx = 0;

                curve.push_back(cv::Point(i, j));
                mean_p.x = i;
                mean_p.y = j;
                curve_idx++;

                int akt_idx = 0;

                while (akt_idx < curve_idx)
                {
                    cv::Point akt_pos = curve[akt_idx];
                    for (int k1 = -1; k1 < 2; k1++)
                        for (int k2 = -1; k2 < 2; k2++)
                        {
                            if (akt_pos.x + k1 >= start_x && akt_pos.x + k1 < end_x && akt_pos.y + k2 >= start_y &&
                                akt_pos.y + k2 < end_y)
                                if (!check[akt_pos.x + k1][akt_pos.y + k2])
                                    if (edge.data[(edge.cols * (akt_pos.y + k2)) + (akt_pos.x + k1)] == 255)
                                    {
                                        check[akt_pos.x + k1][akt_pos.y + k2] = 1;

                                        mean_p.x += akt_pos.x + k1;
                                        mean_p.y += akt_pos.y + k2;
                                        curve.push_back(cv::Point(akt_pos.x + k1, akt_pos.y + k2));
                                        curve_idx++;
                                    }
                        }
                    akt_idx++;
                }

                if (curve_idx > 0 && curve.size() > 0)
                {
                    add_curve = true;
                    mean_p.x = floor((double(mean_p.x) / double(curve_idx)) + 0.5);
                    mean_p.y = floor((double(mean_p.y) / double(curve_idx)) + 0.5);

                    for (int i = 0; i < curve.size(); i++)
                        if (abs(mean_p.x - curve[i].x) <= mean_dist && abs(mean_p.y - curve[i].y) <= mean_dist)
                            add_curve = false;

                    // is ellipse fit possible
                    if (add_curve)
                    {
                        cv::RotatedRect ellipse = cv::fitEllipse(cv::Mat(curve));

                        if (ellipse.center.x < 0 || ellipse.center.y < 0 || ellipse.center.x > pic.cols ||
                            ellipse.center.y > pic.rows)
                        {
                            add_curve = false;
                        }

                        if (ellipse.size.height > 2.0 * ellipse.size.width ||
                            ellipse.size.width > 2.0 * ellipse.size.height)
                        {
                            add_curve = false;
                        }
                    }

                    if (add_curve)
                    {
                        if (inner_color_range > 0)
                        {
                            mean_inner_gray = 0;

                            // calc inner mean
                            for (int i = 0; i < curve.size(); i++)
                            {
                                if (pic.data[(pic.cols * (curve[i].y + 1)) + (curve[i].x)] != 0 ||
                                    pic.data[(pic.cols * (curve[i].y - 1)) + (curve[i].x)] != 0)
                                    if (sqrt(pow(double(curve[i].y - mean_p.y), 2) +
                                             pow(double(curve[i].x - mean_p.x) + 2, 2)) <
                                        sqrt(pow(double(curve[i].y - mean_p.y), 2) +
                                             pow(double(curve[i].x - mean_p.x) - 2, 2)))

                                        mean_inner_gray +=
                                            static_cast<unsigned char>(pic.data[(pic.cols * (curve[i].y)) + (curve[i].x + 2)]);
                                    else
                                        mean_inner_gray +=
                                            static_cast<unsigned char>(pic.data[(pic.cols * (curve[i].y)) + (curve[i].x - 2)]);

                                else if (pic.data[(pic.cols * (curve[i].y)) + (curve[i].x + 1)] != 0 ||
                                         pic.data[(pic.cols * (curve[i].y)) + (curve[i].x - 1)] != 0)
                                    if (sqrt(pow(double(curve[i].y - mean_p.y + 2), 2) +
                                             pow(double(curve[i].x - mean_p.x), 2)) <
                                        sqrt(pow(double(curve[i].y - mean_p.y - 2), 2) +
                                             pow(double(curve[i].x - mean_p.x), 2)))

                                        mean_inner_gray +=
                                            static_cast<unsigned char>(pic.data[(pic.cols * (curve[i].y + 2)) + (curve[i].x)]);
                                    else
                                        mean_inner_gray +=
                                            static_cast<unsigned char>(pic.data[(pic.cols * (curve[i].y - 2)) + (curve[i].x)]);

                                else if (pic.data[(pic.cols * (curve[i].y + 1)) + (curve[i].x + 1)] != 0 ||
                                         pic.data[(pic.cols * (curve[i].y - 1)) + (curve[i].x - 1)] != 0)
                                    if (sqrt(pow(double(curve[i].y - mean_p.y - 2), 2) +
                                             pow(double(curve[i].x - mean_p.x + 2), 2)) <
                                        sqrt(pow(double(curve[i].y - mean_p.y + 2), 2) +
                                             pow(double(curve[i].x - mean_p.x - 2), 2)))

                                        mean_inner_gray +=
                                            static_cast<unsigned char>(pic.data[(pic.cols * (curve[i].y - 2)) + (curve[i].x + 2)]);
                                    else
                                        mean_inner_gray +=
                                            static_cast<unsigned char>(pic.data[(pic.cols * (curve[i].y + 2)) + (curve[i].x - 2)]);

                                else if (pic.data[(pic.cols * (curve[i].y - 1)) + (curve[i].x + 1)] != 0 ||
                                         pic.data[(pic.cols * (curve[i].y + 1)) + (curve[i].x - 1)] != 0)
                                    if (sqrt(pow(double(curve[i].y - mean_p.y + 2), 2) +
                                             pow(double(curve[i].x - mean_p.x + 2), 2)) <
                                        sqrt(pow(double(curve[i].y - mean_p.y - 2), 2) +
                                             pow(double(curve[i].x - mean_p.x - 2), 2)))

                                        mean_inner_gray +=
                                            static_cast<unsigned char>(pic.data[(pic.cols * (curve[i].y + 2)) + (curve[i].x + 2)]);
                                    else
                                        mean_inner_gray +=
                                            static_cast<unsigned char>(pic.data[(pic.cols * (curve[i].y - 2)) + (curve[i].x - 2)]);

                                // mean_inner_gray+=pic.data[( pic.cols*( curve[i].y+((mean_p.y-curve[i].y)/2) ) ) +
                                // ( curve[i].x+((mean_p.x-curve[i].x)/2) )];
                            }

                            mean_inner_gray = floor((double(mean_inner_gray) / double(curve.size())) + 0.5);

                            if (mean_inner_gray_last > (mean_inner_gray + inner_color_range))
                            {
                                mean_inner_gray_last = mean_inner_gray;
                                all_curves.clear();
                                all_curves.push_back(curve);
                            }
                            else if (mean_inner_gray_last <= (mean_inner_gray + inner_color_range) &&
                                     mean_inner_gray_last >= (mean_inner_gray - inner_color_range))
                            {
                                if (curve.size() > all_curves[0].size())
                                {
                                    mean_inner_gray_last = mean_inner_gray;
                                    all_curves.clear();
                                    all_curves.push_back(curve);
                                }
                            }
                        }
                        else
                            all_curves.push_back(curve);
                    }
                }
            }
        }


    return all_curves;
}

cv::RotatedRect find_best_edge(const cv::Mat& pic,
                               const cv::Mat& edge,
                               int start_x,
                               int end_x,
                               int start_y,
                               int end_y,
                               double mean_dist,
                               int inner_color_range)
{
    cv::RotatedRect ellipse;
    ellipse.center.x = 0;
    ellipse.center.y = 0;
    ellipse.angle = 0.0;
    ellipse.size.height = 0.0;
    ellipse.size.width = 0.0;

    std::vector<std::vector<cv::Point>> all_curves =
        get_curves(pic, edge, start_x, end_x, start_y, end_y, mean_dist, inner_color_range);

    if (all_curves.size() == 1)
    {
        ellipse = cv::fitEllipse(cv::Mat(all_curves[0]));

        if (ellipse.center.x < 0 || ellipse.center.y < 0 || ellipse.center.x > pic.cols ||
            ellipse.center.y > pic.rows)
        {
            ellipse.center.x = 0;
            ellipse.center.y = 0;
            ellipse.angle = 0.0;
            ellipse.size.height = 0.0;
            ellipse.size.width = 0.0;
        }
    }
    else
    {
        ellipse.center.x = 0;
        ellipse.center.y = 0;
        ellipse.angle = 0.0;
        ellipse.size.height = 0.0;
        ellipse.size.width = 0.0;
    }

    return ellipse;
}

int calc_pos(int* hist, int mini, int max_region_hole, int min_region_size, int real_hist_sz)
{
    int pos = 0;

    int mean_pos = 0;
    int pos_hole = 0;
    int count = 0;
    int hole_size = 0;
    bool region_start = false;

    for (int i = 0; i < DEF_SIZE; i++)
    {
        if (hist[i] > mini && !region_start)
        {
            region_start = true;
            count++;
            mean_pos += i;
        }
        else if (hist[i] > mini && region_start)
        {
            count += 1 + hole_size;
            mean_pos += i + pos_hole;
            hole_size = 0;
            pos_hole = 0;
        }
        else if (hist[i] <= mini && region_start && hole_size < max_region_hole)
        {
            hole_size++;
            pos_hole += i;
        }
        else if (hist[i] <= mini && region_start && hole_size >= max_region_hole && count >= min_region_size)
        {
            if (count < 1)
                count = 1;
            mean_pos = mean_pos / count;
            if (pow(double((real_hist_sz / 2) - mean_pos), 2) < pow(double((real_hist_sz / 2) - pos), 2))
                pos = mean_pos;

            pos_hole = 0;
            hole_size = 0;
            region_start = 0;
            count = 0;
            mean_pos = 0;
        }
        else if (hist[i] <= mini && region_start && hole_size >= max_region_hole && count < min_region_size)
        {
            pos_hole = 0;
            hole_size = 0;
            region_start = 0;
            count = 0;
            mean_pos = 0;
        }
    }

    return pos;
}

cv::Point th_angular_histo(const cv::Mat& pic,
                           cv::Mat& pic_th,
                           int start_x,
                           int end_x,
                           int start_y,
                           int end_y,
                           int th,
                           double th_histo,
                           int max_region_hole,
                           int min_region_size)
{
    cv::Point pos(0, 0);

    if (start_x < 0)
        start_x = 0;
    if (start_y < 0)
        start_y = 0;
    if (end_x > pic.cols)
        end_x = pic.cols;
    if (end_y > pic.rows)
        end_y = pic.rows;

    int max_l = 0;
    int max_lb = 0;
    int max_b = 0;
    int max_br = 0;

    int min_l, min_lb, min_b, min_br;
    int pos_l, pos_lb, pos_b, pos_br;

    int hist_l[DEF_SIZE];
    int hist_lb[DEF_SIZE];
    int hist_b[DEF_SIZE];
    int hist_br[DEF_SIZE];

    for (int i = 0; i < DEF_SIZE; i++)
    {
        hist_l[i] = 0;
        hist_lb[i] = 0;
        hist_b[i] = 0;
        hist_br[i] = 0;
    }

    int idx_lb = 0;
    int idx_br = 0;

    for (int i = start_x; i < end_x; i++)
    {
        for (int j = start_y; j < end_y; j++)
        {
            if (pic.data[(pic.cols * j) + i] < th)
            {
                pic_th.data[(pic.cols * j) + i] = 255;

                idx_lb = (pic.cols / 2) + (i - (pic.cols / 2)) + (j);
                idx_br = (pic.cols / 2) + (i - (pic.cols / 2)) + (pic.rows - j);

                if (j >= 0 && j < DEF_SIZE && i >= 0 && i < DEF_SIZE && idx_lb >= 0 && idx_lb < DEF_SIZE &&
                    idx_br >= 0 && idx_br < DEF_SIZE)
                {
                    if (++hist_l[j] > max_l)
                        max_l = hist_l[j];

                    if (++hist_b[i] > max_b)
                        max_b = hist_b[i];

                    if (++hist_lb[idx_lb] > max_lb)
                        max_lb = hist_lb[idx_lb];

                    if (++hist_br[idx_br] > max_br)
                        max_br = hist_br[idx_br];
                }
            }
        }
    }

    min_l = max_l - floor(max_l * th_histo);
    min_lb = max_lb - floor(max_lb * th_histo);
    min_b = max_b - floor(max_b * th_histo);
    min_br = max_br - floor(max_br * th_histo);

    pos_l = calc_pos(hist_l, min_l, max_region_hole, min_region_size, pic.rows);
    pos_lb = calc_pos(hist_lb, min_lb, max_region_hole, min_region_size, pic.cols + pic.rows);
    pos_b = calc_pos(hist_b, min_b, max_region_hole, min_region_size, pic.cols);
    pos_br = calc_pos(hist_br, min_br, max_region_hole, min_region_size, pic.cols + pic.rows);


    if (pos_l > 0 && pos_lb > 0 && pos_b > 0 && pos_br > 0)
    {
        pos.x = floor(((pos_b + (floor((((pos_br + pic.rows) - pos_lb) / 2) + 0.5) + pos_lb - pic.rows)) / 2) + 0.5);
        pos.y = floor(((pos_l + (pic.rows - floor((((pos_br + pic.rows) - pos_lb) / 2) + 0.5))) / 2) + 0.5);
    }
    else if (pos_l > 0 && pos_b > 0)
    {
        pos.x = pos_b;
        pos.y = pos_l;
    }
    else if (pos_lb > 0 && pos_br > 0)
    {
        pos.x = floor((((pos_br + pic.rows) - pos_lb) / 2) + 0.5) + pos_lb - pic.rows;
        pos.y = pic.rows - floor((((pos_br + pic.rows) - pos_lb) / 2) + 0.5);
    }

    if (pos.x < 0)
        pos.x = 0;
    if (pos.y < 0)
        pos.y = 0;
    if (pos.x >= pic.cols)
        pos.x = 0;
    if (pos.y >= pic.rows)
        pos.y = 0;


    return pos;
}

void grow_region(cv::RotatedRect& ellipse, const cv::Mat& pic, int max_ellipse_radi)
{
    float mean = 0.0;

    int x0 = ellipse.center.x;
    int y0 = ellipse.center.y;

    int mini = 1000;
    int maxi = 0;

    for (int i = -2; i < 3; i++)
        for (int j = -2; j < 3; j++)
        {
            if (y0 + j > 0 && y0 + j < pic.rows && x0 + i > 0 && x0 + i < pic.cols)
            {
                if (mini > pic.data[(pic.cols * (y0 + j)) + (x0 + i)])
                    mini = pic.data[(pic.cols * (y0 + j)) + (x0 + i)];

                if (maxi < pic.data[(pic.cols * (y0 + j)) + (x0 + i)])
                    maxi = pic.data[(pic.cols * (y0 + j)) + (x0 + i)];

                mean += pic.data[(pic.cols * (y0 + j)) + (x0 + i)];
            }
        }

    mean = mean / 25.0;

    float diff = abs(mean - pic.data[(pic.cols * (y0)) + (x0)]);

    int th_up = ceil(mean + diff) + 1;
    int th_down = floor(mean - diff) - 1;

    int radi = 0;

    for (int i = 1; i < max_ellipse_radi; i++)
    {
        radi = i;

        int left = 0;
        int right = 0;
        int top = 0;
        int bottom = 0;

        for (int j = -i; j <= 1 + (i * 2); j++)
        {
            // left
            if (y0 + j > 0 && y0 + j < pic.rows && x0 + i > 0 && x0 + i < pic.cols)
                if (pic.data[(pic.cols * (y0 + j)) + (x0 + i)] > th_down &&
                    pic.data[(pic.cols * (y0 + j)) + (x0 + i)] < th_up)
                {
                    left++;
                    // pic.data[(pic.cols*(y0+j))+(x0+i)]=255;
                }

            // right
            if (y0 + j > 0 && y0 + j < pic.rows && x0 - i > 0 && x0 - i < pic.cols)
                if (pic.data[(pic.cols * (y0 + j)) + (x0 - i)] > th_down &&
                    pic.data[(pic.cols * (y0 + j)) + (x0 - i)] < th_up)
                {
                    right++;
                    // pic.data[(pic.cols*(y0+j))+(x0-i)]=255;
                }

            // top
            if (y0 - i > 0 && y0 - i < pic.rows && x0 + j > 0 && x0 + j < pic.cols)
                if (pic.data[(pic.cols * (y0 - i)) + (x0 + j)] > th_down &&
                    pic.data[(pic.cols * (y0 - i)) + (x0 + j)] < th_up)
                {
                    top++;
                    // pic.data[(pic.cols*(y0-i))+(x0+j)]=255;
                }

            // bottom
            if (y0 + i > 0 && y0 + i < pic.rows && x0 + j > 0 && x0 + j < pic.cols)
                if (pic.data[(pic.cols * (y0 + i)) + (x0 + j)] > th_down &&
                    pic.data[(pic.cols * (y0 + i)) + (x0 + j)] < th_up)
                {
                    bottom++;
                    // pic.data[(pic.cols*(y0+i))+(x0+j)]=255;
                }
        }

        float p_left = float(left) / float(1 + (2 * i));
        float p_right = float(right) / float(1 + (2 * i));
        float p_top = float(top) / float(1 + (2 * i));
        float p_bottom = float(bottom) / float(1 + (2 * i));

        if (p_top < 0.2 && p_bottom < 0.2)
            break;

        if (p_left < 0.2 && p_right < 0.2)
            break;
    }

    ellipse.size.height = radi;
    ellipse.size.width = radi;

}

bool is_good_ellipse(cv::RotatedRect& ellipse, const cv::Mat& pic, int good_ellipse_threshold, int max_ellipse_radi)
{
    if (ellipse.center.x == 0 && ellipse.center.y == 0)
        return false;

    if (ellipse.size.width == 0 || ellipse.size.height == 0)
        grow_region(ellipse, pic, max_ellipse_radi);

    const float x0 = ellipse.center.x;
    const float y0 = ellipse.center.y;

    const cheshm::PixelBox inner = {
        static_cast<int>(x0 - ellipse.size.width / 4.0),
        static_cast<int>(x0 + ellipse.size.width / 4.0),
        static_cast<int>(y0 - ellipse.size.height / 4.0),
        static_cast<int>(y0 + ellipse.size.height / 4.0),
    };
    const cheshm::PixelBox outer = {
        static_cast<int>(x0 - ellipse.size.width * 0.75),
        static_cast<int>(x0 + ellipse.size.width * 0.75),
        static_cast<int>(y0 - ellipse.size.height * 0.75),
        static_cast<int>(y0 + ellipse.size.height * 0.75),
    };
    const cheshm::PixelBox cutout = {
        static_cast<int>(x0 - ellipse.size.width / 2),
        static_cast<int>(x0 + ellipse.size.width / 2) + 1,
        static_cast<int>(y0 - ellipse.size.height / 2),
        static_cast<int>(y0 + ellipse.size.height / 2) + 1,
    };

    return cheshm::check_ellipse_intensity_gap(pic, inner, outer, cutout, static_cast<float>(good_ellipse_threshold))
        .passes;
}

void rays(const cv::Mat& th_edges, int end_x, int end_y, const cv::Point& pos, int (&ret)[8])
{
    for (int i = 0; i < 8; i++)
        ret[i] = -1;
    for (int i = 0; i < end_x; i++)
        for (int j = 0; j < end_y; j++)
        {
            if (pos.x - i > 0 && pos.x + i < th_edges.cols && pos.y - j > 0 && pos.y + j < th_edges.rows)
            {
                if (static_cast<int>(th_edges.data[(th_edges.cols * (pos.y)) + (pos.x + i)]) != 0 && ret[0] == -1)
                {
                    ret[0] = th_edges.data[(th_edges.cols * (pos.y)) + (pos.x + i)] - 1;
                }
                if (static_cast<int>(th_edges.data[(th_edges.cols * (pos.y)) + (pos.x - i)]) != 0 && ret[1] == -1)
                {
                    ret[1] = th_edges.data[(th_edges.cols * (pos.y)) + (pos.x - i)] - 1;
                }
                if (static_cast<int>(th_edges.data[(th_edges.cols * (pos.y + j)) + (pos.x)]) != 0 && ret[2] == -1)
                {
                    ret[2] = th_edges.data[(th_edges.cols * (pos.y + j)) + (pos.x)] - 1;
                }
                if (static_cast<int>(th_edges.data[(th_edges.cols * (pos.y - j)) + (pos.x)]) != 0 && ret[3] == -1)
                {
                    ret[3] = th_edges.data[(th_edges.cols * (pos.y - j)) + (pos.x)] - 1;
                }

                if (static_cast<int>(th_edges.data[(th_edges.cols * (pos.y + j)) + (pos.x + i)]) != 0 && ret[4] == -1 && i == j)
                {
                    ret[4] = th_edges.data[(th_edges.cols * (pos.y + j)) + (pos.x + i)] - 1;
                }
                if (static_cast<int>(th_edges.data[(th_edges.cols * (pos.y - j)) + (pos.x - i)]) != 0 && ret[5] == -1 && i == j)
                {
                    ret[5] = th_edges.data[(th_edges.cols * (pos.y - j)) + (pos.x - i)] - 1;
                }
                if (static_cast<int>(th_edges.data[(th_edges.cols * (pos.y - j)) + (pos.x + i)]) != 0 && ret[6] == -1 && i == j)
                {
                    ret[6] = th_edges.data[(th_edges.cols * (pos.y - j)) + (pos.x + i)] - 1;
                }
                if (static_cast<int>(th_edges.data[(th_edges.cols * (pos.y + j)) + (pos.x - i)]) != 0 && ret[7] == -1 && i == j)
                {
                    ret[7] = th_edges.data[(th_edges.cols * (pos.y + j)) + (pos.x - i)] - 1;
                }
            }
        }
}

void zero_around_region_th_border(const cv::Mat& pic,
                                  cv::Mat& edges,
                                  cv::Mat& th_edges,
                                  int th,
                                  int edge_to_th,
                                  double mean_dist,
                                  double area,
                                  cv::RotatedRect& pos)
{
    int ret[8];
    std::vector<cv::Point> selected_points;
    cv::RotatedRect ellipse;

    int start_x = pos.center.x - (area * pic.cols);
    int end_x = pos.center.x + (area * pic.cols);
    int start_y = pos.center.y - (area * pic.rows);
    int end_y = pos.center.y + (area * pic.rows);

    if (start_x < 0)
        start_x = edge_to_th;
    if (start_y < 0)
        start_y = edge_to_th;
    if (end_x > pic.cols)
        end_x = pic.cols - (edge_to_th + 1);
    if (end_y > pic.rows)
        end_y = pic.rows - (edge_to_th + 1);

    th = th + th + 1;

    for (int i = start_x; i < end_x; i++)
        for (int j = start_y; j < end_y; j++)
        {
            if (pic.data[(pic.cols * j) + (i)] < th)
            {
                for (int k1 = -edge_to_th; k1 < edge_to_th; k1++)
                    for (int k2 = -edge_to_th; k2 < edge_to_th; k2++)
                    {
                        if (i + k1 >= 0 && i + k1 < pic.cols && j + k2 > 0 && j + k2 < edges.rows)
                            if (static_cast<int>(edges.data[(edges.cols * (j + k2)) + (i + k1)]))
                                th_edges.data[(edges.cols * (j + k2)) + (i + k1)] = 255;
                    }
            }
        }

    // remove_points_with_low_angle(th_edges, start_x, end_x, start_y, end_y);
    std::vector<std::vector<cv::Point>> all_curves =
        get_curves(pic, th_edges, start_x, end_x, start_y, end_y, mean_dist, 0);


    if (all_curves.size() > 0)
    {
        // zero th_edges

        for (int i = 0; i < th_edges.cols; i++)
            for (int j = 0; j < th_edges.rows; j++)
            {
                th_edges.data[(th_edges.cols * (j)) + (i)] = 0;
            }

        // draw remaining edges
        for (int i = 0; i < all_curves.size(); i++)
        {
            for (int j = 0; j < all_curves[i].size(); j++)
            {
                if (all_curves[i][j].x >= 0 && all_curves[i][j].x < th_edges.cols && all_curves[i][j].y >= 0 &&
                    all_curves[i][j].y < th_edges.rows)
                    th_edges.data[(th_edges.cols * (all_curves[i][j].y)) + (all_curves[i][j].x)] =
                        i + 1; //+1 becouse of first is 0
            }
        }

        cv::Point st_pos;
        st_pos.x = pos.center.x;
        st_pos.y = pos.center.y;
        // send rays add edges to vector
        rays(th_edges, (end_x - start_x) / 2, (end_y - start_y) / 2, st_pos, ret);


        // gather points
        for (int i = 0; i < 8; i++)
            if (ret[i] > -1 && ret[i] < all_curves.size())
            {
                for (int j = 0; j < all_curves[ret[i]].size(); j++)
                {
                    selected_points.push_back(all_curves[ret[i]][j]);
                }
            }
        // ellipse fit if size>5

        if (selected_points.size() > 5)
        {
            pos = cv::fitEllipse(cv::Mat(selected_points));
        }
    }

}

void optimize_pos(const cv::Mat& pic, double area, cv::Point& pos)
{
    int start_x = pos.x - (area * pic.cols);
    int end_x = pos.x + (area * pic.cols);
    int start_y = pos.y - (area * pic.rows);
    int end_y = pos.y + (area * pic.rows);

    int val;
    int min_akt;
    int min_val = 1000000;

    int pos_x = 0;
    int pos_y = 0;
    int pos_count = 0;

    int reg_size = sqrt(sqrt(pow(double(area * pic.cols * 2), 2) + pow(double(area * pic.rows * 2), 2)));

    if (start_x < reg_size)
        start_x = reg_size;
    if (start_y < reg_size)
        start_y = reg_size;
    if (end_x > pic.cols)
        end_x = pic.cols - (reg_size + 1);
    if (end_y > pic.rows)
        end_y = pic.rows - (reg_size + 1);

    for (int i = start_x; i < end_x; i++)
        for (int j = start_y; j < end_y; j++)
        {
            min_akt = 0;

            for (int k1 = -reg_size; k1 < reg_size; k1++)
                for (int k2 = -reg_size; k2 < reg_size; k2++)
                {
                    if (i + k1 > 0 && i + k1 < pic.cols && j + k2 > 0 && j + k2 < pic.rows)
                    {
                        val = (pic.data[(pic.cols * j) + (i)] - pic.data[(pic.cols * (j + k2)) + (i + k1)]);
                        if (val > 0)
                            min_akt += val;
                    }
                }

            if (min_akt == min_val)
            {
                pos_x += i;
                pos_y += j;
                pos_count++;
            }

            if (min_akt < min_val)
            {
                min_val = min_akt;
                pos_x = i;
                pos_y = j;
                pos_count = 1;
            }
        }

    if (pos_count > 0)
    {
        pos.x = pos_x / pos_count;
        pos.y = pos_y / pos_count;
    }
}

std::optional<cv::RotatedRect>
runexcuse(cv::Mat& pic, cv::Mat& pic_th, cv::Mat& th_edges, int good_ellipse_threshold, int max_ellipse_radi)
{
    // mean under mean
    // mean_under_mean(pic, 5);
    cv::normalize(pic, pic, 0, 255, cv::NORM_MINMAX, CV_8U);

    double border = 0.1;
    int peek_detector_factor = 10;
    int bright_region_th = 199;
    double mean_dist = 3;
    int inner_color_range = 5;
    double th_histo = 0.5;
    int max_region_hole = 5;
    int min_region_size = 7;
    double area_opt = 0.1;
    double area_edges = 0.2;
    int edge_to_th = 5;

    cv::RotatedRect ellipse;
    cv::Point pos(0, 0);

    int start_x = floor(double(pic.cols) * border);
    int start_y = floor(double(pic.rows) * border);

    int end_x = pic.cols - start_x;
    int end_y = pic.rows - start_y;

    double stddev = 0;
    bool edges_only_tried = false;
    bool peek_found = false;
    int threshold_up = 0;

    // cvtColor(*m, pic, CV_RGB2GRAY);

    peek_found = peek(pic, stddev, start_x, end_x, start_y, end_y, peek_detector_factor, bright_region_th);
    threshold_up = ceil(stddev / 2);
    threshold_up--;

    cv::Mat picpic = cv::Mat::zeros(end_y - start_y, end_x - start_x, CV_8U);

    for (int i = 0; i < picpic.cols; i++)
        for (int j = 0; j < picpic.rows; j++)
        {
            picpic.data[(picpic.cols * j) + i] = pic.data[(pic.cols * (start_y + j)) + (start_x + i)];
        }

    // Canny( detected_edges2, detected_edges2, stddev*0.4, stddev, 3 );
    const int non_edge_pixel_count = static_cast<int>(0.7 * picpic.cols * picpic.rows);
    cv::Mat detected_edges2 = cheshm::canny_gaussian16(picpic, non_edge_pixel_count, 64);

    cv::Mat detected_edges = cv::Mat::zeros(pic.rows, pic.cols, CV_8U);
    for (int i = 0; i < detected_edges2.cols; i++)
        for (int j = 0; j < detected_edges2.rows; j++)
        {
            detected_edges.data[(detected_edges.cols * (start_y + j)) + (start_x + i)] =
                detected_edges2.data[(detected_edges2.cols * j) + i];
        }

    remove_points_with_low_angle(detected_edges, start_x, end_x, start_y, end_y);

    // peek_found=1;
    if (peek_found)
    {
        edges_only_tried = true;
        ellipse = find_best_edge(pic, detected_edges, start_x, end_x, start_y, end_y, mean_dist, inner_color_range);

        if (ellipse.center.x <= 0 || ellipse.center.x >= pic.cols || ellipse.center.y <= 0 ||
            ellipse.center.y >= pic.rows)
        {
            ellipse.center.x = 0;
            ellipse.center.y = 0;
            ellipse.angle = 0.0;
            ellipse.size.height = 0.0;
            ellipse.size.width = 0.0;
            peek_found = false;
        }
    }

    if (!peek_found)
    {
        pos = th_angular_histo(
            pic, pic_th, start_x, end_x, start_y, end_y, threshold_up, th_histo, max_region_hole, min_region_size);

        ellipse.center.x = pos.x;
        ellipse.center.y = pos.y;
        ellipse.angle = 0.0;
        ellipse.size.height = 0.0;
        ellipse.size.width = 0.0;
    }

    if (pos.x == 0 && pos.y == 0 && !edges_only_tried)
    {
        ellipse = find_best_edge(pic, detected_edges, start_x, end_x, start_y, end_y, mean_dist, inner_color_range);
        peek_found = true;
    }

    if (pos.x > 0 && pos.y > 0 && pos.x < pic.cols && pos.y < pic.rows && !peek_found)
    {
        optimize_pos(pic, area_opt, pos);

        ellipse.center.x = pos.x;
        ellipse.center.y = pos.y;
        ellipse.angle = 0.0;
        ellipse.size.height = 0.0;
        ellipse.size.width = 0.0;
        zero_around_region_th_border(
            pic, detected_edges, th_edges, threshold_up, edge_to_th, mean_dist, area_edges, ellipse);
    }


    if (is_good_ellipse(ellipse, pic, good_ellipse_threshold, max_ellipse_radi))
        return ellipse;
    return std::nullopt;
}

} // namespace

std::optional<cv::RotatedRect>
findPupilEllipse(const cv::Mat& frame, int max_ellipse_radi, int good_ellipse_threshold)
{
    cv::Mat downscaled = frame;
    float scalingRatio = 1.0f;
    if (frame.rows > IMG_SIZE || frame.cols > IMG_SIZE)
    {
        const float rw = static_cast<float>(IMG_SIZE) / static_cast<float>(frame.cols);
        const float rh = static_cast<float>(IMG_SIZE) / static_cast<float>(frame.rows);
        scalingRatio = std::min<float>(std::min<float>(rw, rh), 1.0f);
        cv::resize(frame, downscaled, cv::Size(), scalingRatio, scalingRatio, cv::INTER_LINEAR);
    }

    cv::Mat target;
    cv::normalize(downscaled, target, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat pic_th = cv::Mat::zeros(target.rows, target.cols, CV_8U);
    cv::Mat th_edges = cv::Mat::zeros(target.rows, target.cols, CV_8U);

    const auto ellipse = runexcuse(target, pic_th, th_edges, good_ellipse_threshold, max_ellipse_radi);
    if (!ellipse)
        return std::nullopt;
    return cv::RotatedRect(cv::Point2f(ellipse->center.x / scalingRatio, ellipse->center.y / scalingRatio),
                           cv::Size2f(ellipse->size.width / scalingRatio, ellipse->size.height / scalingRatio),
                           ellipse->angle);
}

} // namespace cheshm::ExCuSe
