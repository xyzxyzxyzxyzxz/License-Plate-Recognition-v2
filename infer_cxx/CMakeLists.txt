cmake_minimum_required(VERSION 3.20)
project(infer_cxx)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(onnxruntimesetup)

set(OpenCV_DIR "C:/opencv/opencv/build")
find_package(OpenCV REQUIRED)

add_executable(infer_lp_cpp main.cpp utils.cpp "coord_lp.h" "coord_lp.cpp" "proc_img.cpp")

set_target_properties(infer_lp_cpp PROPERTIES CXX_STANDARD 20 CXX_EXTENSIONS OFF)
target_link_libraries(infer_lp_cpp PRIVATE onnxruntime_interface)
target_link_libraries(infer_lp_cpp PRIVATE ${OpenCV_LIBS})

target_include_directories(infer_lp_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})

set(YOLO_ONNX "yolo.onnx")
set(PROVINCE_RESNET_ONNX "province.onnx")
set(AZ01_RESNET_ONNX "az01.onnx")

copy_file_to_bin_dir(${YOLO_ONNX})
copy_file_to_bin_dir(${PROVINCE_RESNET_ONNX})
copy_file_to_bin_dir(${AZ01_RESNET_ONNX})
copy_file_to_bin_dir(Input.jpg)

target_compile_definitions(infer_lp_cpp PRIVATE -DYOLO_ONNX_FILE="${YOLO_ONNX}" -DPROVINCE_RESNET_ONNX_FILE="${PROVINCE_RESNET_ONNX}" -DAZ01_RESNET_ONNX_FILE="${AZ01_RESNET_ONNX}")

set_target_properties(infer_lp_cpp
	PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
)