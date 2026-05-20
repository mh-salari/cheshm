cmake_minimum_required(VERSION 3.25)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
  set(CMAKE_C_FLAGS_RELEASE "-O3 ${CMAKE_C_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS_RELEASE}")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV is needed by C++ detectors that wrap upstream pupil-tracking
# kernels (Starburst, Swirski2D, etc.). Headers come via `opencv_core`
# and `opencv_imgproc`; .dylib/.so/.dll linked per-detector.
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_subdirectory(src/lavan/limbus_detectors/daugman)
add_subdirectory(src/lavan/pupil_detectors/starburst)
