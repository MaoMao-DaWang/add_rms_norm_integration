cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "Ascend architecture")
find_package(ASC REQUIRED)
project(add_rms_norm_integration LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

function(query_python output_var expression)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "${expression}"
        OUTPUT_VARIABLE value OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "Python query failed: ${expression}")
    endif()
    string(REPLACE "\r" "\n" lines "${value}")
    string(REGEX MATCH "[^\n]+$" last_line "${lines}")
    set(${output_var} "${last_line}" PARENT_SCOPE)
endfunction()
query_python(TORCH_PREFIX "import torch; print(torch.utils.cmake_prefix_path)")
find_package(Torch REQUIRED HINTS ${TORCH_PREFIX})
query_python(TORCH_NPU_DIR "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).parent)")
find_path(CANN_INCLUDE_DIR acl/acl.h
    HINTS "$ENV{ASCEND_HOME_PATH}" "${ASCEND_CANN_PACKAGE_PATH}"
    PATH_SUFFIXES include aarch64-linux/include)
find_library(ASCENDCL_LIBRARY ascendcl
    HINTS "$ENV{ASCEND_HOME_PATH}" "${ASCEND_CANN_PACKAGE_PATH}"
    PATH_SUFFIXES lib64 aarch64-linux/lib64)
if(NOT CANN_INCLUDE_DIR OR NOT ASCENDCL_LIBRARY)
    message(FATAL_ERROR "Cannot find CANN ACL headers/library; check ASCEND_HOME_PATH")
endif()

add_library(arm_kernels SHARED kernel.asc)
target_compile_options(arm_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
target_include_directories(arm_kernels PRIVATE ${CANN_INCLUDE_DIR})
target_link_libraries(arm_kernels PRIVATE ${ASCENDCL_LIBRARY})
find_file(ASCENDC_RUNTIME libascendc_runtime.a
    HINTS "$ENV{ASCEND_HOME_PATH}" "${ASCEND_CANN_PACKAGE_PATH}"
    PATH_SUFFIXES lib64 aarch64-linux/lib64)
if(ASCENDC_RUNTIME)
    target_link_libraries(arm_kernels PRIVATE ${ASCENDC_RUNTIME})
endif()

add_library(arm_torch SHARED torch_binding.cpp)
target_include_directories(arm_torch PRIVATE ${TORCH_INCLUDE_DIRS} ${TORCH_NPU_DIR}/include ${CANN_INCLUDE_DIR})
target_link_directories(arm_torch PRIVATE ${TORCH_NPU_DIR}/lib)
separate_arguments(TORCH_COMPILE_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(arm_torch PRIVATE ${TORCH_COMPILE_FLAGS})
target_link_libraries(arm_torch PRIVATE arm_kernels ${TORCH_LIBRARIES} torch_npu ${ASCENDCL_LIBRARY} Python3::Python)
set_target_properties(arm_torch PROPERTIES BUILD_RPATH "$ORIGIN;${TORCH_NPU_DIR}/lib")
