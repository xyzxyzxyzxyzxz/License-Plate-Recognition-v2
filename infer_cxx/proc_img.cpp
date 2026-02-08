#include "proc_img.h"
#include <opencv2/opencv.hpp>



int split_one_lp(const cv::Mat& img, const std::vector<float>& bbox, cv::Mat& return_lp, std::vector<std::array<float,4>>& return_recs) {
	std::array<float, 8> coord14 = get_lp_x1y1_to_x4y4(bbox);
	std::array<float, 4> coord_lp13 = get_lp_x1y1_x3y3(coord14);
	std::array<float, 8> coord_lp_cut14 = get_lp_cut_x1y1_to_x4y4(coord14, coord_lp13);

	auto [lp_x1, lp_y1, lp_x3, lp_y3] = coord_lp13;
	
	cv::Mat lp = img(cv::Rect(static_cast<int>(lp_x1), static_cast<int>(lp_y1), static_cast<int>(lp_x3)- static_cast<int>(lp_x1)+1, static_cast<int>(lp_y3)- static_cast<int>(lp_y1)+1));
	
	auto [lp_cut_x1, lp_cut_y1, lp_cut_x2, lp_cut_y2, lp_cut_x3, lp_cut_y3, lp_cut_x4, lp_cut_y4] = coord_lp_cut14;
	
	std::array<cv::Point2f,4> src_pts{
		cv::Point2f{ lp_cut_x1,lp_cut_y1 },
		cv::Point2f{ lp_cut_x4,lp_cut_y4 },
		cv::Point2f{ lp_cut_x3,lp_cut_y3 },
		cv::Point2f{ lp_cut_x2,lp_cut_y2 }
	};

	float w = lp_x3 - lp_x1;
	float h = lp_y3 - lp_y1;

	std::array<cv::Point2f, 4> dst_pts{
		cv::Point2f{ 0.0f,0.0f },
		cv::Point2f{ w ,0.0f },
		cv::Point2f{ w ,h },
		cv::Point2f{ 0.0f,h }
	};

	cv::Mat to_M = cv::getPerspectiveTransform(src_pts.data(), dst_pts.data());
	cv::Mat to_lp;
	cv::warpPerspective(lp,to_lp, to_M, cv::Size(static_cast<int>(w), static_cast<int>(h)));

	//车牌标准440x140
	cv::resize(to_lp, to_lp, cv::Size(880, 280), 0, 0, cv::INTER_LINEAR);
	to_lp = to_lp(cv::Rect(cv::Point(15,50),cv::Point(880-35, 280-55)));

	return_lp = to_lp.clone();

	cv::GaussianBlur(to_lp, to_lp, cv::Size(5, 5), 0);

	//直方图均衡化
	to_lp = equalize_hist_color(to_lp);
	cv::imshow("equalize hist", to_lp);

	//提取蓝底白字
	cv::Scalar range_l(0, 0, 120);
	cv::Scalar range_u(179, 110, 255);
	cv::Mat to_lp_hsv;
	cv::cvtColor(to_lp, to_lp_hsv, cv::COLOR_BGR2HSV);
	cv::Mat white_alpha;
	cv::inRange(to_lp_hsv, range_l, range_u, white_alpha);

	cv::imshow("white alpha", white_alpha);

	cv::Mat h_ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 180));
	cv::Mat h_ker1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 11));
	cv::Mat w_ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
	cv::Mat white_alpha_o;
	cv::erode(white_alpha, white_alpha_o, h_ker1);
	cv::dilate(white_alpha_o, white_alpha_o, h_ker);
	cv::dilate(white_alpha_o, white_alpha_o, w_ker);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(white_alpha_o, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	cv::Mat d_c;
	cv::cvtColor(white_alpha_o, d_c, cv::COLOR_GRAY2BGR);
	cv::drawContours(d_c, contours, -1, cv::Scalar(0, 0, 255), 1);
	cv::imshow("draw c", d_c);

	std::vector<std::array<float, 4>> cnt_apl_rec;
	for (const auto& cnt : contours) {
		cv::Rect rec = cv::boundingRect(cnt);
		if (rec.width>15&&rec.height>150&&rec.width<150) {
			cnt_apl_rec.push_back({ static_cast<float>(rec.x), static_cast<float>(rec.y), static_cast<float>(rec.width), static_cast<float>(rec.height) });
		}
	}

	std::sort(cnt_apl_rec.begin(), cnt_apl_rec.end(), [](const std::array<float, 4>& a, const std::array<float, 4>& b) {
		return a[0] < b[0];
		});

	if (cnt_apl_rec.size() == 8 &&
		(cnt_apl_rec[1][0] - (cnt_apl_rec[0][0] + cnt_apl_rec[0][2])) < 20 &&
		cnt_apl_rec[0][2] < 60
		) {
		return_recs.push_back({ cnt_apl_rec[0][0],cnt_apl_rec[0][1],
							cnt_apl_rec[1][0] + cnt_apl_rec[1][2] - cnt_apl_rec[0][0],
							cnt_apl_rec[1][1] + cnt_apl_rec[1][3] - cnt_apl_rec[0][1] });
		for (size_t i = 2; i < cnt_apl_rec.size(); ++i) {
			return_recs.push_back(cnt_apl_rec[i]);
		}
	}
	else {
		return_recs = cnt_apl_rec;
	}

	if (return_recs.size() == 8) {
		return_recs.erase(return_recs.begin() + 1);
	}

	cv::waitKey(0);

	return 0;

}

cv::Mat equalize_hist_color(const cv::Mat& img) {
	cv::Mat ycrcb_img;

	cv::cvtColor(img, ycrcb_img, cv::COLOR_BGR2YCrCb);

	std::vector<cv::Mat> channels;
	cv::split(ycrcb_img, channels);

	cv::equalizeHist(channels[0], channels[0]);
	cv::merge(channels, ycrcb_img);

	cv::Mat result_img;
	cv::cvtColor(ycrcb_img, result_img, cv::COLOR_YCrCb2BGR);

	return result_img;

}

int show_lp_rec(const cv::Mat& lp_pers,const std::vector<std::array<float, 4>>& lp_rec) {
	cv::Mat vis = lp_pers.clone();
	for (auto rec : lp_rec) {
		cv::rectangle(vis, cv::Rect(static_cast<int>(rec[0]), static_cast<int>(rec[1]), static_cast<int>(rec[2]), static_cast<int>(rec[3])), cv::Scalar(0, 0, 255), 1);
	}
	cv::imshow("lp_rec", vis);
	cv::waitKey(0);

	return 0;
}

std::vector<float> img_to_float(const cv::Mat& img_bgr) {
	cv::Mat rgb;

	cv::cvtColor(img_bgr, rgb, cv::COLOR_BGR2RGB);

	cv::Mat float_img;
	rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

	std::vector<float> img_f_vec(1 * 3 * float_img.cols * float_img.rows);

	for (int y = 0; y < float_img.rows; y++) {
		const cv::Vec3f* row_ptr = float_img.ptr<cv::Vec3f>(y);
		for (int x = 0; x < float_img.cols; x++) {
			float r = row_ptr[x][0];
			float g = row_ptr[x][1];
			float b = row_ptr[x][2];

			//r = (r - kMean[0]) / kStd[0];
			//g = (g - kMean[1]) / kStd[1];
			//b = (b - kMean[2]) / kStd[2];

			img_f_vec[x + y * float_img.cols + 0 * float_img.cols * float_img.rows] = r;
			img_f_vec[x + y * float_img.cols + 1 * float_img.cols * float_img.rows] = g;
			img_f_vec[x + y * float_img.cols + 2 * float_img.cols * float_img.rows] = b;
		}
	}

	return img_f_vec;


}

const cv::Size alpha_size(90, 180);

std::vector<float> get_province_img(const cv::Mat& lp, std::vector<std::array<float, 4>>& recs, const std::array<float, 3>& mean, const std::array<float, 3>& std) {
	std::array<float, 4> rec = recs[0];
	cv::Mat province_img = lp(cv::Rect(static_cast<int>(rec[0]), static_cast<int>(rec[1]), static_cast<int>(rec[2]), static_cast<int>(rec[3])));
	cv::resize(province_img, province_img, alpha_size, 0, 0, cv::INTER_LINEAR);

	province_img.convertTo(province_img, CV_32FC3, 1.0 / 255.0);
	cv::cvtColor(province_img, province_img, cv::COLOR_BGR2RGB);

	std::vector<cv::Mat> channels(3);
	cv::split(province_img, channels);

	for (int i = 0; i < 3; i++) {
		channels[i] = (channels[i] - mean[i]) / std[i];
	}
	std::vector<float> province_f_vec(alpha_size.width * alpha_size.height * 3);

	for (int c = 0; c < 3; c++) {
		for (int y = 0; y < alpha_size.height; y++) {
			const float* row_ptr = channels[c].ptr<float>(y);
			for (int x = 0; x < alpha_size.width; x++) {
				province_f_vec[x + y * alpha_size.width + c * alpha_size.width * alpha_size.height] = row_ptr[x];
			}
		}
	}

	return province_f_vec;
}

std::vector<float> get_az01_img(const cv::Mat& lp, std::vector<std::array<float, 4>>& recs, const std::array<float, 3>& mean, const std::array<float, 3>& std) {
	std::vector<float> az01_f_vec;
	az01_f_vec.resize((recs.size() - 1) * alpha_size.width * alpha_size.height * 3);
	for (int i = 1; i < recs.size(); i++) {
		std::array<float, 4> rec = recs[i];
		cv::Mat az01_img = lp(cv::Rect(static_cast<int>(rec[0]), static_cast<int>(rec[1]), static_cast<int>(rec[2]), static_cast<int>(rec[3])));
		cv::resize(az01_img, az01_img, alpha_size, 0, 0, cv::INTER_LINEAR);

		az01_img.convertTo(az01_img, CV_32FC3, 1.0 / 255.0);
		cv::cvtColor(az01_img, az01_img, cv::COLOR_BGR2RGB);

		std::vector<cv::Mat> channels(3);
		cv::split(az01_img, channels);

		for (int i = 0; i < 3; i++) {
			channels[i] = (channels[i] - mean[i]) / std[i];
		}

		for (int c = 0; c < 3; c++) {
			for (int y = 0; y < alpha_size.height; y++) {
				const float* row_ptr = channels[c].ptr<float>(y);
				for (int x = 0; x < alpha_size.width; x++) {
					az01_f_vec[(i - 1) * alpha_size.width * alpha_size.height * 3 + x + y * alpha_size.width + c * alpha_size.width * alpha_size.height] = row_ptr[x];
				}
			}
		}
	}

	return az01_f_vec;
}