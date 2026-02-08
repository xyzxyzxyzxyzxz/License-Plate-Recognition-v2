#pragma once

#include <vector>
#include <array>

std::array<float,8> get_lp_x1y1_to_x4y4(const std::vector<float>& one_bbox);
std::array<float, 4> get_lp_x1y1_x3y3(const std::array<float, 8>& coord);
std::array<float, 8> get_lp_cut_x1y1_to_x4y4(const std::array<float, 8>& coord, const std::array<float, 4>& coord_lp);