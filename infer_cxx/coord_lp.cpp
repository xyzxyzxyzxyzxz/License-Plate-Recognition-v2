#include "coord_lp.h"

#include <stdexcept>

#define OUTPUT_DIM_3 14

std::array<float, 8> get_lp_x1y1_to_x4y4(const std::vector<float>& one_bbox) {
	std::array<float, 8> coord14{};

	if (one_bbox.size() != OUTPUT_DIM_3) {
		throw std::runtime_error("one_bbox size is not 14");
	}
	std::copy_n(one_bbox.begin() + 6, 8, coord14.begin());

	return coord14;
}

std::array<float,4> get_lp_x1y1_x3y3(const std::array<float,8>& coord) {
	std::array<float, 4> coord_lp{};

	coord_lp[0]=(std::min(coord[0], coord[2]));
	coord_lp[1]=(std::min(coord[1], coord[7]));
	coord_lp[2]=(std::max(coord[4], coord[6]));
	coord_lp[3]=(std::max(coord[3], coord[5]));

	return coord_lp;
}

std::array<float, 8> get_lp_cut_x1y1_to_x4y4(const std::array<float, 8>& coord, const std::array<float, 4>& coord_lp) {
	auto [x1, y1, x2, y2, x3, y3, x4, y4] = coord;
	auto [pl_x1, pl_y1, pl_x3, pl_y3] = coord_lp;

	float pl_cut_x1 = x1 - pl_x1;
	float pl_cut_y1 = y1 - pl_y1;
	float pl_cut_x3 = x3 - pl_x1;
	float pl_cut_y3 = y3 - pl_y1;
	float pl_cut_x2 = x2 - pl_x1;
	float pl_cut_y2 = y2 - pl_y1;
	float pl_cut_x4 = x4 - pl_x1;
	float pl_cut_y4 = y4 - pl_y1;

	return { pl_cut_x1, pl_cut_y1,
		pl_cut_x2, pl_cut_y2,
		pl_cut_x3, pl_cut_y3,
		pl_cut_x4, pl_cut_y4 };
}