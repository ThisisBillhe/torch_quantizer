cmake_minimum_required(VERSION 3.11)
project(torch_quantizer LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)           # -pthread
find_package(OpenMP REQUIRED)            # -Xpreprocessor -fopenmp
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC
set(CMAKE_CXX_VISIBILITY_PRESET hidden)  # -fvisibility=hidden

if (MSVC)
    string(APPEND CMAKE_CXX_FLAGS " /Wall")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " /Zi")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " /O2 /Ob2")
else ()
    string(APPEND CMAKE_CXX_FLAGS " -Wall")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g -Og")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
endif ()

set(USE_FP16 ON)

if (NOT DEFINED USE_FP16 AND NOT "$ENV{USE_FP16}" STREQUAL "")
    set(USE_FP16 "$ENV{USE_FP16}")
endif ()

if (NOT DEFINED USE_FP16)
    set(USE_FP16 OFF)
    message(WARNING "FP16 support disabled, compiling without torch.HalfTensor. Suppress this warning with -DUSE_FP16=ON or -DUSE_FP16=OFF.")
elseif (USE_FP16)
    message(STATUS "FP16 support enabled, compiling with torch.HalfTensor.")
else ()
    message(STATUS "FP16 support disabled, compiling without torch.HalfTensor.")
endif ()

if (USE_FP16)
    add_definitions(-DUSE_FP16)
endif ()

find_package(CUDA REQUIRED)
if (CUDA_FOUND AND NOT WIN32)
    message(STATUS "Found CUDA, enabling CUDA support.")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD "${CMAKE_CXX_STANDARD}")
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_definitions(-D__USE_CUDA__)

    string(APPEND CMAKE_CUDA_FLAGS " $ENV{TORCH_NVCC_FLAGS}")

    if (NOT DEFINED TORCH_CUDA_ARCH_LIST AND NOT "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
        set(TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
    endif ()

    if (NOT TORCH_CUDA_ARCH_LIST)
        set(TORCH_CUDA_ARCH_LIST "Auto")
        message(WARNING "Torch CUDA arch list is not set, setting to \"Auto\". Suppress this warning with -DTORCH_CUDA_ARCH_LIST=Common.")
    endif ()

    set(CMAKE_CUDA_ARCHITECTURES OFF)
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${TORCH_CUDA_ARCH_LIST})
    message(STATUS "TORCH_CUDA_ARCH_LIST: \"${TORCH_CUDA_ARCH_LIST}\"")
    message(STATUS "CUDA_ARCH_FLAGS: \"${CUDA_ARCH_FLAGS}\"")
    list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr" "--expt-extended-lambda")
    if (CUDA_HAS_FP16 OR NOT "${CUDA_VERSION}" VERSION_LESS "7.5")
        if (USE_FP16)
            message(STATUS "Found CUDA with FP16 support, compiling with torch.cuda.HalfTensor.")
            string(APPEND CMAKE_CUDA_FLAGS " -DCUDA_HAS_FP16=1"
                    " -D__CUDA_NO_HALF_OPERATORS__"
                    " -D__CUDA_NO_HALF_CONVERSIONS__"
                    " -D__CUDA_NO_HALF2_OPERATORS__"
                    " -D__CUDA_NO_BFLOAT16_CONVERSIONS__")
        else ()
            message(STATUS "Found CUDA with FP16 support, but it is suppressed by the compile options, compiling without torch.cuda.HalfTensor.")
        endif ()
    else ()
        message(STATUS "Could not find CUDA with FP16 support, compiling without torch.cuda.HalfTensor.")
    endif ()

    foreach (FLAG ${CUDA_NVCC_FLAGS})
        string(FIND "${FLAG}" " " flag_space_position)
        if (NOT flag_space_position EQUAL -1)
            message(FATAL_ERROR "Found spaces in CUDA_NVCC_FLAGS entry '${FLAG}'")
        endif ()
        string(APPEND CMAKE_CUDA_FLAGS " ${FLAG}")
    endforeach ()
    string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
    message(STATUS "CMAKE_CUDA_FLAGS: \"${CMAKE_CUDA_FLAGS}\"")

    if (MSVC)
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} /O2 /Ob2")
    else ()
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
    endif ()
elseif (NOT CUDA_FOUND)
    message(STATUS "CUDA not found, build for CPU-only.")
else ()
    message(STATUS "CUDA found, but build for CPU-only on Windows.")
endif ()

function(system)
    set(options STRIP)
    set(oneValueArgs OUTPUT_VARIABLE ERROR_VARIABLE WORKING_DIRECTORY)
    set(multiValueArgs COMMAND)
    cmake_parse_arguments(
            SYSTEM
            "${options}"
            "${oneValueArgs}"
            "${multiValueArgs}"
            "${ARGN}"
    )

    if (NOT DEFINED SYSTEM_WORKING_DIRECTORY)
        set(SYSTEM_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif ()

    execute_process(
            COMMAND ${SYSTEM_COMMAND}
            OUTPUT_VARIABLE STDOUT
            ERROR_VARIABLE STDERR
            WORKING_DIRECTORY "${SYSTEM_WORKING_DIRECTORY}"
    )

    if ("${SYSTEM_STRIP}")
        string(STRIP "${STDOUT}" STDOUT)
        string(STRIP "${STDERR}" STDERR)
    endif ()

    set("${SYSTEM_OUTPUT_VARIABLE}" "${STDOUT}" PARENT_SCOPE)

    if (DEFINED SYSTEM_ERROR_VARIABLE)
        set("${SYSTEM_ERROR_VARIABLE}" "${STDERR}" PARENT_SCOPE)
    endif ()
endfunction()

if (NOT DEFINED PYTHON_EXECUTABLE)
    if (WIN32)
        set(PYTHON_EXECUTABLE "python.exe")
    else ()
        set(PYTHON_EXECUTABLE "python")
    endif ()
endif ()

if (UNIX)
    system(
            STRIP OUTPUT_VARIABLE PYTHON_EXECUTABLE
            COMMAND bash -c "type -P '${PYTHON_EXECUTABLE}'"
    )
endif ()

system(
        STRIP OUTPUT_VARIABLE PYTHON_VERSION
        COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('platform').python_version())"
)

message(STATUS "Use Python version: ${PYTHON_VERSION}")
message(STATUS "Use Python executable: \"${PYTHON_EXECUTABLE}\"")

if (NOT DEFINED PYTHON_INCLUDE_DIR)
    message(STATUS "Auto detecting Python include directory...")
    system(
            STRIP OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
            COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('sysconfig').get_path('platinclude'))"
    )
endif ()

if ("${PYTHON_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "Python include directory not found")
else ()
    message(STATUS "Detected Python include directory: \"${PYTHON_INCLUDE_DIR}\"")
    include_directories("${PYTHON_INCLUDE_DIR}")
endif ()

system(
        STRIP OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('sysconfig').get_path('purelib'))"
)
message(STATUS "Detected Python site packages: \"${PYTHON_SITE_PACKAGES}\"")

find_package(Git REQUIRED)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    message(STATUS "Populating Git submodule.")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR 
                "git submodule updata --init --recursive failed with ${GIT_SUBMOD_RESULT}.")
    endif()
endif()

if (NOT DEFINED TORCH_INCLUDE_PATH)
    message(STATUS "Auto detecting Torch include directory...")
    system(
            STRIP OUTPUT_VARIABLE TORCH_INCLUDE_PATH
            COMMAND "${PYTHON_EXECUTABLE}" -c "print('\\\;'.join(__import__('torch.utils.cpp_extension', fromlist=[None]).include_paths()))"
    )

    if ("${TORCH_INCLUDE_PATH}" STREQUAL "")
        set(TORCH_INCLUDE_PATH "${PYTHON_SITE_PACKAGES}/torch/include")
    endif ()
endif ()

if ("${TORCH_INCLUDE_PATH}" STREQUAL "")
    message(FATAL_ERROR "Torch include directory not found. Got: \"${TORCH_INCLUDE_PATH}\"")
else ()
    message(STATUS "Detected Torch include directory: \"${TORCH_INCLUDE_PATH}\"")
    include_directories(${TORCH_INCLUDE_PATH})
endif ()

if (NOT DEFINED TORCH_LIBRARY_PATH)
    message(STATUS "Auto detecting Torch library directory...")
    system(
            STRIP OUTPUT_VARIABLE TORCH_LIBRARY_PATH
            COMMAND "${PYTHON_EXECUTABLE}" -c "print('\\\;'.join(__import__('torch.utils.cpp_extension', fromlist=[None]).library_paths()))"
    )

    if ("${TORCH_LIBRARY_PATH}" STREQUAL "")
        set(TORCH_LIBRARY_PATH "${PYTHON_SITE_PACKAGES}/torch/lib")
    endif ()
endif ()

if ("${TORCH_LIBRARY_PATH}" STREQUAL "")
    message(FATAL_ERROR "Torch library directory not found. Got: \"${TORCH_LIBRARY_PATH}\"")
else ()
    message(STATUS "Detected Torch library directory: \"${TORCH_LIBRARY_PATH}\"")
endif ()

unset(TORCH_LIBRARIES)

foreach (VAR_PATH ${TORCH_LIBRARY_PATH})
    file(GLOB TORCH_LIBRARY "${VAR_PATH}/*")
    message(STATUS "Detected Torch libraries: \"${TORCH_LIBRARY}\"")
endforeach ()

foreach (VAR_PATH ${TORCH_LIBRARY_PATH})
    if (WIN32)
        file(GLOB TORCH_LIBRARY "${VAR_PATH}/*.lib")
    else ()
        file(GLOB TORCH_LIBRARY "${VAR_PATH}/libtorch_python.*")
    endif ()
    list(APPEND TORCH_LIBRARIES "${TORCH_LIBRARY}")
endforeach ()

message(STATUS "Detected Torch Python libraries: \"${TORCH_LIBRARIES}\"")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ERROR)

# Set the desired flags for CUTLASS
set(CUTLASS_NVCC_ARCHS 80 CACHE STRING "Set CUDA architectures for CUTLASS")
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Disable CUTLASS tests")
set(CUTLASS_UNITY_BUILD_ENABLED ON CACHE BOOL "Enable CUTLASS Unity Build")

add_subdirectory(third-party/cutlass)
add_subdirectory(third-party/pybind11)
add_subdirectory(third-party/googletest)
set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})

include_directories("${CMAKE_SOURCE_DIR}")
include_directories(include)
add_subdirectory(src)
