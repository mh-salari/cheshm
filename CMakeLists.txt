cmake_minimum_required(VERSION 3.24)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE)
  # OpenCV 4 expects c++1z on the macOS clang used by recent SDKs.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
endif()

if(MSVC)
  # M_PI + aligned-storage workaround mirroring upstream pupil-detectors.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES -D_ENABLE_EXTENDED_ALIGNED_STORAGE")
endif()

# Speed flags — pupil-detectors recommends -O2 (not -O3) for its C++ kernels.
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 ${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_C_FLAGS_RELEASE "-O3 ${CMAKE_C_FLAGS_RELEASE}")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# Plain-C kernels loaded at runtime via ctypes (no Cython, no Python.h).
add_subdirectory(src/lavan/boundary)

# Cython + C++ Pupil-Labs 2D detector, vendored under LGPL v3.
add_subdirectory(src/lavan/pupil_detector_2d)
