//#include <onnxruntime/core/graph/constants.h>
#include "constants.h"
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/onnxruntime_c_api.h>
#include <onnxruntime/onnxruntime_run_options_config_keys.h>
#include <onnxruntime/onnxruntime_session_options_config_keys.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <regex>
#include <vector>

#include "argparsing.h"

#include "utils.h"
#include "proc_img.h"

#include <opencv2/opencv.hpp>

#include <chrono>

#if ORT_API_VERSION < 23
#error "Onnx runtime header too old. Version >=1.23.0 assumed"
#endif


cv::Size YOLO_IMAGE_SIZE(928,928);
#ifndef OUTPUT_DIM_3
#define OUTPUT_DIM_3 14
#endif // !OUTP

//const int OUTPUT_DIM_3 = 14;



using OrtFileString = std::basic_string<ORTCHAR_T>;

static OrtFileString toOrtFileString(const std::filesystem::path& path) {
    std::string string(path.string());
    return { string.begin(), string.end() };
}

#define PROVIDER_LIB_PAIR(NAME) \
  std::pair { NAME, DLL_NAME("onnxruntime_providers_" NAME) }


static void register_execution_providers(Ort::Env& env) {
    // clang-format off
    std::array provider_libraries{
        //PROVIDER_LIB_PAIR("nv_tensorrt_rtx"),
        PROVIDER_LIB_PAIR("cuda"),
        PROVIDER_LIB_PAIR("openvino"),
        PROVIDER_LIB_PAIR("qnn"),
        PROVIDER_LIB_PAIR("cann"),
    };

    // clang-format on

    for (auto& [registration_name, dll] : provider_libraries) {
        auto providers_library = get_executable_path().parent_path() / dll;
        if (!std::filesystem::is_regular_file(providers_library)) {
            LOG("{} does not exist! Skipping execution provider", providers_library.string());
            continue;
        }
        try {
            env.RegisterExecutionProviderLibrary(registration_name, toOrtFileString(providers_library));
        }
        catch (std::exception& ex) {
            LOG("Failed to register {}! Skipping execution provider", providers_library.string());
        }
    }
}


Ort::ConstMemoryInfo match_common_memory_info(const Ort::Session& input_session, const Ort::Session& output_session) {
    auto input_infos = input_session.GetMemoryInfoForOutputs();
    auto output_infos = output_session.GetMemoryInfoForInputs();

    // First try to find a common non-CPU allocator
    for (auto& in : input_infos) {
        for (auto& out : output_infos) {
            if (in == out && in.GetDeviceType() != OrtMemoryInfoDeviceType_CPU &&
                in.GetDeviceMemoryType() == OrtDeviceMemoryType_DEFAULT) {
                return in;
            }
        }
    }
    // If impossible then also allow to fall back to CPU
    for (auto& in : input_infos) {
        for (auto& out : output_infos) {
            if (in == out) {
                return in;
            }
        }
    }
    THROW_ERROR("Could not find a common allocator");
}



static Ort::SessionOptions create_session_options(Ort::Env& env, const Opts& opts) {
    std::vector<Ort::ConstEpDevice> selected_devices;
    auto ep_devices = env.GetEpDevices();
    LOG("{} devices found", ep_devices.size());
    for (auto& device : ep_devices) {
        auto metadata = device.Device().Metadata();
        // LUID can be used on Windows platform to match EpDevices with
        // IDXGIAdapter in case an application already has a device selection
        // logic based on `IDXGIAdapter`s
        auto luid = metadata.GetValue("LUID");
        LOG("Vendor: {}, EpName: {}, DeviceId: 0x{:x}, LUID: {}, Vendor: {}", device.EpVendor(), device.EpName(),
            device.Device().DeviceId(), luid ? luid : "<unavailable>", device.Device().Vendor());
        if (to_uppercase(opts.select_vendor) == to_uppercase(device.Device().Vendor())) {
            selected_devices.push_back(device);

            LOG("++++++++++Vendor: {}, EpName: {}, DeviceId: 0x{:x}, LUID: {}", device.EpVendor(), device.EpName(),
                device.Device().DeviceId(), luid ? luid : "<unavailable>");
        }
        //if (to_uppercase(opts.select_ep) == to_uppercase(device.EpName())) {
        //    selected_devices.push_back(device);
        //    LOG("++++++++++Vendor: {}, EpName: {}, DeviceId: 0x{:x}, LUID: {}", device.EpVendor(), device.EpName(),
        //        device.Device().DeviceId(), luid ? luid : "<unavailable>");
        //}
    }

    Ort::SessionOptions so;
    if (!selected_devices.empty()) {
        Ort::KeyValuePairs ep_options;
        // Select EP for manually selected devices
        so.AppendExecutionProvider_V2(env, selected_devices, ep_options);
    }

    so.SetEpSelectionPolicy(opts.ep_device_policy);
    return so;
}


static Ort::Session create_session(Ort::Env& env, std::filesystem::path& model_file,
    const Ort::SessionOptions& session_options) {
    if (!std::filesystem::is_regular_file(model_file)) {
        THROW_ERROR("Model file \"{}\" does not exist!", model_file.string());
    }
    try {
        Ort::Session session(env, toOrtFileString(model_file).c_str(), session_options);
        return session;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[ORT] create_session failed: " << e.what() << std::endl;
        throw;
    }
}

//output: x1 y1 x2 y2 conf class kpx1 kpy1 ...
int get_bbox_kp_vec(const float* data,const std::vector<int64_t>& shape,std::vector<std::vector<float>>& bbox,const float conf_thr) {
    for (int b = 0; b < shape[0]; b++) {
        for (int n_b = 0; n_b < shape[1]; n_b++) {
            //std::vector<int> bbox_1;
            auto box_start = data + b * shape[1] * shape[2] + n_b * shape[2];
            //std::cout <<"n: "<<n_b << "==============conf: " << *(box_start + 4) << std::endl;
            if (*(box_start + 4) > conf_thr) {
                bbox.emplace_back(box_start, box_start + shape[2]);
            }
            
        }
    }

    return 0;
}

cv::Mat letter_box(cv::Mat img_src,cv::Size o_size) {
    //cv::Mat img_o;
    //cv::resize(img_src, img_o, o_size, 0, 0, cv::INTER_LINEAR);
    //return img_o;
    const int src_w = img_src.cols, src_h = img_src.rows;
    const int new_w = o_size.width, new_h = o_size.height;
    const float r = std::min(static_cast<float>(new_w) / src_w, static_cast<float>(new_h) / src_h);

    std::pair<float, float> ratio = { r,r };
    std::pair<float, float> new_unpad = { std::round(src_w * r),std::round(src_h * r) };
    float dw = new_w - new_unpad.first, dh = new_h - new_unpad.second;

    dw /= 2;
    dh /= 2;
    cv::Mat img_resize;
    cv::resize(img_src, img_resize, cv::Size(new_unpad.first, new_unpad.second), 0, 0, cv::INTER_LINEAR);
    float top = std::round(dh - 0.1), bottom = std::round(dh + 0.1);
    float left = std::round(dw - 0.1), right = std::round(dw + 0.1);
    cv::Mat img_border;
    cv::copyMakeBorder(img_resize, img_border, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return img_border;
}

int scale_boxes(cv::Size img,std::vector<std::vector<float>>& pred,cv::Size orig_img) {
    float gain = std::min(static_cast<float>(img.height) / orig_img.height, static_cast<float>(img.width) / orig_img.width);
    float pad_x = std::round((img.width - orig_img.width * gain)/2 - 0.1);
    float pad_y = std::round((img.height - orig_img.height * gain) / 2 - 0.1);

    for (int i = 0; i < pred.size(); i++) {
        pred[i][0] = (pred[i][0] - pad_x)/gain;
        pred[i][1] = (pred[i][1] - pad_y)/gain;
        pred[i][2] = (pred[i][2] - pad_x)/gain;
        pred[i][3] = (pred[i][3] - pad_y)/gain;

        pred[i][0] = std::clamp(pred[i][0], 0.f, static_cast<float>(orig_img.width));
        pred[i][1] = std::clamp(pred[i][1], 0.f, static_cast<float>(orig_img.height));
        pred[i][2] = std::clamp(pred[i][2], 0.f, static_cast<float>(orig_img.width));
        pred[i][3] = std::clamp(pred[i][3], 0.f, static_cast<float>(orig_img.height));

    }

    return 0;
}

int scale_coords(cv::Size img, std::vector<std::vector<float>>& pred, cv::Size orig_img) {
    float gain = std::min(static_cast<float>(img.height) / orig_img.height, static_cast<float>(img.width) / orig_img.width);
    std::pair<float, float> pad = { (img.width - orig_img.width * gain) / 2,(img.height - orig_img.height * gain) / 2 };

    for (int i = 0; i < pred.size(); i++) {
        for (int kp_i = 6; kp_i < OUTPUT_DIM_3; kp_i += 2) {
            pred[i][kp_i] = (pred[i][kp_i] - pad.first)/gain;
            pred[i][kp_i + 1] = (pred[i][kp_i + 1] - pad.second)/gain;

            pred[i][kp_i] = std::clamp(pred[i][kp_i], 0.f, static_cast<float>(orig_img.width));
            pred[i][kp_i+1] = std::clamp(pred[i][kp_i+1], 0.f, static_cast<float>(orig_img.height));

        }
    }

    return 0;
}


std::vector<size_t> argmax(const float* pred, const std::vector<int64_t>& shape) {
	std::vector<size_t> max_idx(shape[0], 0);
    for (int i = 0; i < shape[0]; i++) {
		max_idx[i] = std::distance(pred + i * shape[1], std::max_element(pred + i * shape[1], pred + (i + 1) * shape[1]));
    }

	return max_idx;
}

std::vector<std::string> province_a_l = { "Íî", "»¦", "½ò", "Óå", "¼½", "½ú", "ÃÉ", "ÁÉ", "¼ª", "ºÚ", "ËÕ", "Õã", "¾©", "Ãö", "¸Ó", "Â³", "Ô¥", "¶õ", "Ïæ", "ÔÁ", "¹ð", "Çí", "´¨", "¹ó", "ÔÆ", "²Ø", "ÉÂ", "¸Ê", "Çà", "Äþ", "ÐÂ", "¾¯", "Ñ§", "O" };
std::vector<char> az01_a_l = { 'A', 'B', 'C', 'D', 'E',
'F', 'G', 'H', 'J', 'K',
'L', 'M', 'N', 'P', 'Q',
'R', 'S', 'T', 'U', 'V',
'W', 'X', 'Y', 'Z', '0',
'1', '2', '3', '4', '5',
'6', '7', '8', '9', 'O' };

auto main(int argc, char** argv) -> int {
    try {
        Opts opts = parse_args(argc, argv);
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);


        auto version_string = Ort::GetVersionString();
        auto build_info = api->GetBuildInfoString();

        LOG("Hello from ONNX runtime version: {} (build info {})\n", version_string, build_info);

        // Setup ORT environment
        auto env = Ort::Env(ORT_LOGGING_LEVEL_WARNING);
        register_execution_providers(env);
        // Create session options for ORT environment according to command line
        // parameters
        auto session_options = create_session_options(env, opts);

        // Load a ONNX files
        std::string model_file = YOLO_ONNX_FILE;
        auto model_path = get_executable_path().parent_path() / YOLO_ONNX_FILE;  // defined via CMAKE
        auto model_context_file = std::regex_replace(model_file, std::regex(".onnx$"), "_ctx.onnx");
        auto model_context_path = get_executable_path().parent_path() / model_context_file;
        bool use_model_context = std::filesystem::is_regular_file(model_context_path);
        auto load_path = use_model_context ? model_context_path : model_path;



        //TODO: LetterBox
        std::cout << "file name :" << opts.input_image.c_str() << std::endl;
        cv::Mat img_mat = cv::imread(opts.input_image.c_str());

		cv::Mat img_lp_ori = img_mat.clone();
        
        cv::Mat img_letter_box = letter_box(img_mat,YOLO_IMAGE_SIZE);
        int64_t size_cv = static_cast<int64_t>(img_letter_box.cols);
        auto input_img_data = img_to_float(img_letter_box);

        std::vector<int64_t> input_shape_cv{ 1,3,size_cv,size_cv };


        CHECK_ORT(api->AddFreeDimensionOverrideByName(session_options, "N", 1));
        CHECK_ORT(api->AddFreeDimensionOverrideByName(session_options, "W", size_cv));
        CHECK_ORT(api->AddFreeDimensionOverrideByName(session_options, "H", size_cv));
        if (!use_model_context) {
            session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, opts.enableEpContext ? "1" : "0");
        }

        auto infer_session = create_session(env, load_path, session_options);

        Ort::AllocatorWithDefaultOptions cpu_allocator;
        
        auto output_shape = infer_session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        Ort::Value input_value = Ort::Value::CreateTensor<float>(cpu_allocator.GetInfo(), input_img_data.data(), input_img_data.size(), input_shape_cv.data(), input_shape_cv.size());
        Ort::Value output_value = Ort::Value::CreateTensor<float>(cpu_allocator, output_shape.data(), output_shape.size());

        Ort::IoBinding inference_binding(infer_session);

        try {
            inference_binding.BindInput("images", input_value);
        }
        catch (const Ort::Exception& e) {
            std::cerr << "BindInput failed: " << e.what()
                << " | code=" << e.GetOrtErrorCode() << "\n";
            throw;
        }
        inference_binding.BindOutput("output0", output_value);

        Ort::RunOptions run_options;
        run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");

        for (int i = 0; i < 1; i++) {
            //auto start_t = std::chrono::steady_clock::now();
            double t_s = cv::getTickCount();
            try {
                infer_session.Run(run_options, inference_binding);
            }
            catch (const Ort::Exception& e) {
                std::cerr << "Run(IoBinding) failed: " << e.what()
                    << " | code=" << e.GetOrtErrorCode() << "\n";
                throw;
            }

            //auto end_t = std::chrono::steady_clock::now();
            double t_i_e = cv::getTickCount();

            std::cout << "use time: " << (t_i_e - t_s) / cv::getTickFrequency() * 1000 << std::endl;
        }
        //auto start_t = std::chrono::steady_clock::now();
        double t_s = cv::getTickCount();
        try {
            infer_session.Run(run_options, inference_binding);
        }
        catch (const Ort::Exception& e) {
            std::cerr << "Run(IoBinding) failed: " << e.what()
                << " | code=" << e.GetOrtErrorCode() << "\n";
            throw;
        }

        //auto end_t = std::chrono::steady_clock::now();
        double t_i_e = cv::getTickCount();

        std::cout << "use time: " << (t_i_e - t_s) / cv::getTickFrequency() * 1000 << std::endl;
        //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t);
        //std::cout << "use time c: " << duration.count() << " ms\n";


        inference_binding.SynchronizeOutputs();

        auto output_data = output_value.GetTensorData<float>();


        std::vector<std::vector<float>> bbox;
        if (get_bbox_kp_vec(output_data, output_shape, bbox, 0.6)) {
            std::cout << "get_bbox_kp_vec error" << std::endl;
        }
        
        if (scale_boxes(img_letter_box.size(), bbox, img_mat.size())) {
            std::cout << "scale_boxes error" << std::endl;
        }
        if (scale_coords(img_letter_box.size(), bbox, img_mat.size())) {
            std::cout << "scale_coords error" << std::endl;
        }


        cv::Mat show_lp=img_mat.clone();


        for (int i = 0; i < bbox.size(); i++) {
            std::cout << "the " << i<<" ";

            for (int j = 0; j < bbox[i].size(); j++) {
                std::cout << bbox[i][j]<<" ";
            }
            std::cout<<std::endl;
            int lp_x1, lp_y1, lp_x3, lp_y3;
            lp_x1 = static_cast<int>(bbox[i][0]) - 1;
            lp_y1 = static_cast<int>(bbox[i][1]) - 1;
            lp_x3 = static_cast<int>(bbox[i][2]) + 1;
            lp_y3 = static_cast<int>(bbox[i][3]) + 1;
            std::cout << "lp x1y1x3y3: " << lp_x1 << " " << lp_y1 << " " << lp_x3 << " " << lp_y3 << std::endl;

            cv::Rect lp(cv::Point(lp_x1,lp_y1), cv::Point(lp_x3,lp_y3));
            cv::rectangle(show_lp,lp,cv::Scalar(0,0,255));

            std::vector<int> kpxy;
            for (int kp_i = 6; kp_i < OUTPUT_DIM_3; kp_i++) {
                kpxy.push_back(cvRound(bbox[i][kp_i]));
            }

            for (int kp_i = 0; kp_i < kpxy.size(); kp_i+=2) {
                std::cout << "key point: " << kpxy[kp_i] << " " << kpxy[kp_i + 1] << std::endl;
                cv::circle(show_lp, cv::Point(kpxy[kp_i], kpxy[kp_i + 1]), 4, cv::Scalar(0, 255, 0));
            }


        }

        cv::imshow("lp", show_lp);
        cv::waitKey(0);
        

        cv::Mat return_lp;
        std::vector<std::array<float, 4>> return_recs;
        for (int i = 0; i < bbox.size(); i++) {
			return_recs.clear();
            split_one_lp(img_lp_ori, bbox[i], return_lp, return_recs);
            show_lp_rec(return_lp,return_recs);


            if (return_recs.size() != 7) {
                return EXIT_FAILURE;
            }

            std::array<float, 3> mean = { 0.485f, 0.456f, 0.406f };
            std::array<float, 3> std = { 0.229f, 0.224f, 0.225f };
            auto province_img_data = get_province_img(return_lp, return_recs, mean, std);
            auto az01_img_data = get_az01_img(return_lp, return_recs, mean, std);

            auto session_options_p = create_session_options(env, opts);
            std::string model_file_p = PROVINCE_RESNET_ONNX_FILE;
            auto model_path_p = get_executable_path().parent_path() / PROVINCE_RESNET_ONNX_FILE;  // defined via CMAKE
            auto model_context_file_p = std::regex_replace(model_file_p, std::regex(".onnx$"), "_ctx.onnx");
            auto model_context_path_p = get_executable_path().parent_path() / model_context_file_p;
            bool use_model_context_p = std::filesystem::is_regular_file(model_context_path_p);
            auto load_path_p = use_model_context_p ? model_context_path_p : model_path_p;

            cv::Size alpha_size(90, 180);
            std::vector<int64_t> input_shape_cv_p{ 1,3,alpha_size.height,alpha_size.width };

            //CHECK_ORT(api->AddFreeDimensionOverrideByName(session_options_p, "batch_size", 1));
            //CHECK_ORT(api->AddFreeDimensionOverrideByName(session_options_p, "W", alpha_size.width));
            //CHECK_ORT(api->AddFreeDimensionOverrideByName(session_options_p, "H", alpha_size.height));

            if (!use_model_context) {
                session_options_p.AddConfigEntry(kOrtSessionOptionEpContextEnable, opts.enableEpContext ? "1" : "0");
            }

            auto infer_session_p = create_session(env, load_path_p, session_options_p);

            auto output_shape_p = infer_session_p.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

            Ort::Value input_value_p = Ort::Value::CreateTensor<float>(cpu_allocator.GetInfo(), province_img_data.data(), province_img_data.size(), input_shape_cv_p.data(), input_shape_cv_p.size());
            Ort::Value output_value_p = Ort::Value::CreateTensor<float>(cpu_allocator, output_shape_p.data(), output_shape_p.size());

            Ort::IoBinding inference_binding_p(infer_session_p);

            try {
                inference_binding_p.BindInput("input", input_value_p);
            }
            catch (const Ort::Exception& e) {
                std::cerr << "BindInput failed: " << e.what()
                    << " | code=" << e.GetOrtErrorCode() << "\n";
                throw;
            }
            inference_binding_p.BindOutput("output", output_value_p);

            Ort::RunOptions run_options_p;
            run_options_p.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");

            for (int i = 0; i < 1; i++) {
                t_s = cv::getTickCount();
                try {
                    infer_session_p.Run(run_options_p, inference_binding_p);
                }
                catch (const Ort::Exception& e) {
                    std::cerr << "Run(IoBinding) failed: " << e.what()
                        << " | code=" << e.GetOrtErrorCode() << "\n";
                    throw;
                }

                //auto end_t = std::chrono::steady_clock::now();
                t_i_e = cv::getTickCount();

                std::cout << "use time: " << (t_i_e - t_s) / cv::getTickFrequency() * 1000 << std::endl;
            }

            t_s = cv::getTickCount();
            try {
                infer_session_p.Run(run_options_p, inference_binding_p);
            }
            catch (const Ort::Exception& e) {
                std::cerr << "Run(IoBinding) failed: " << e.what()
                    << " | code=" << e.GetOrtErrorCode() << "\n";
                throw;
            }

            //auto end_t = std::chrono::steady_clock::now();
            t_i_e = cv::getTickCount();

            std::cout << "use time: " << (t_i_e - t_s) / cv::getTickFrequency() * 1000 << std::endl;
            //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t);
            //std::cout << "use time c: " << duration.count() << " ms\n";


            inference_binding_p.SynchronizeOutputs();

            auto output_data_p = output_value_p.GetTensorData<float>();

            auto province_pred_alpha = argmax(output_data_p, output_shape_p);

            for (int i = 0; i < province_pred_alpha.size(); i++) {
                std::cout << "the " << i << " province pred alpha: " << province_a_l[province_pred_alpha[i]] << std::endl;
            }

            //az01
            auto session_options_a = create_session_options(env, opts);
            std::string model_file_a = AZ01_RESNET_ONNX_FILE;
            auto model_path_a = get_executable_path().parent_path() / AZ01_RESNET_ONNX_FILE;  // defined via CMAKE
            auto model_context_file_a = std::regex_replace(model_file_a, std::regex(".onnx$"), "_ctx.onnx");
            auto model_context_path_a = get_executable_path().parent_path() / model_context_file_a;
            bool use_model_context_a = std::filesystem::is_regular_file(model_context_path_a);
            auto load_path_a = use_model_context_a ? model_context_path_a : model_path_a;

            //cv::Size alpha_size(90, 180);
            int az01_num = 6;
            std::vector<int64_t> input_shape_cv_a{ az01_num,3,alpha_size.height,alpha_size.width };

            //CHECK_ORT(api->AddFreeDimensionOverrideByName(session_options_a, "batch_size", 6));
            //CHECK_ORT(api->AddFreeDimensionOverrideByName(session_options_a, "W", alpha_size.width));
            //CHECK_ORT(api->AddFreeDimensionOverrideByName(session_options_a, "H", alpha_size.height));

            if (!use_model_context) {
                session_options_a.AddConfigEntry(kOrtSessionOptionEpContextEnable, opts.enableEpContext ? "1" : "0");
            }

            auto infer_session_a = create_session(env, load_path_a, session_options_a);

            auto output_shape_a = infer_session_a.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

            Ort::Value input_value_a = Ort::Value::CreateTensor<float>(cpu_allocator.GetInfo(), az01_img_data.data(), az01_img_data.size(), input_shape_cv_a.data(), input_shape_cv_a.size());
            Ort::Value output_value_a = Ort::Value::CreateTensor<float>(cpu_allocator, output_shape_a.data(), output_shape_a.size());

            //std::cout << "output shape a" << output_shape_a[0] << std::endl;

            Ort::IoBinding inference_binding_a(infer_session_a);

            try {
                inference_binding_a.BindInput("input", input_value_a);
            }
            catch (const Ort::Exception& e) {
                std::cerr << "BindInput failed: " << e.what()
                    << " | code=" << e.GetOrtErrorCode() << "\n";
                throw;
            }
            inference_binding_a.BindOutput("output", output_value_a);

            Ort::RunOptions run_options_a;
            run_options_a.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");

            for (int i = 0; i < 1; i++) {
                t_s = cv::getTickCount();
                try {
                    infer_session_a.Run(run_options_a, inference_binding_a);
                }
                catch (const Ort::Exception& e) {
                    std::cerr << "Run(IoBinding) failed: " << e.what()
                        << " | code=" << e.GetOrtErrorCode() << "\n";
                    throw;
                }

                //auto end_t = std::chrono::steady_clock::now();
                t_i_e = cv::getTickCount();

                std::cout << "use time: " << (t_i_e - t_s) / cv::getTickFrequency() * 1000 << std::endl;
            }

            t_s = cv::getTickCount();
            try {
                infer_session_a.Run(run_options_a, inference_binding_a);
            }
            catch (const Ort::Exception& e) {
                std::cerr << "Run(IoBinding) failed: " << e.what()
                    << " | code=" << e.GetOrtErrorCode() << "\n";
                throw;
            }

            //auto end_t = std::chrono::steady_clock::now();
            t_i_e = cv::getTickCount();

            std::cout << "use time: " << (t_i_e - t_s) / cv::getTickFrequency() * 1000 << std::endl;
            //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t);
            //std::cout << "use time c: " << duration.count() << " ms\n";


            inference_binding_a.SynchronizeOutputs();

            auto output_data_a = output_value_a.GetTensorData<float>();

            auto az01_pred_alpha = argmax(output_data_a, output_shape_a);

            std::cout << "az01 pred alpha: ";
            for (int i = 0; i < az01_pred_alpha.size(); i++) {
                std::cout << az01_a_l[az01_pred_alpha[i]];
            }

            std::cout << std::endl;
        }






    }catch (const std::runtime_error& err)
    {
        std::cerr << err.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}