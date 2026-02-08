#pragma once

#include <opencv2/core.hpp>
#include <array>
#include "coord_lp.h"
#include <algorithm>

int split_one_lp(const cv::Mat& lp, const std::vector<float>& bbox, cv::Mat& return_lp, std::vector<std::array<float, 4>>& return_recs);
cv::Mat equalize_hist_color(const cv::Mat& img);
int show_lp_rec(const cv::Mat& lp_pers,const std::vector<std::array<float, 4>>& lp_rec);
std::vector<float> img_to_float(const cv::Mat& img_bgr);
std::vector<float> get_province_img(const cv::Mat& lp, std::vector<std::array<float, 4>>& recs, const std::array<float, 3>& mean, const std::array<float, 3>& std);
std::vector<float> get_az01_img(const cv::Mat& lp, std::vector<std::array<float, 4>>& recs, const std::array<float, 3>& mean, const std::array<float, 3>& std);