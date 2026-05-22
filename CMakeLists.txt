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

# nanobind for C++ → Python bindings. Located via the installed
# nanobind package's helper command (works under scikit-build-core
# because nanobind is in build-system.requires).
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

add_library(cheshm_cpp INTERFACE)
add_library(cheshm::cpp ALIAS cheshm_cpp)
target_include_directories(cheshm_cpp INTERFACE include)

add_subdirectory(bindings/python/src/align)
add_subdirectory(bindings/python/src/viz)
add_subdirectory(bindings/python/src/limbus/Daugman)
add_subdirectory(bindings/python/src/pupil/ElSe)
add_subdirectory(bindings/python/src/pupil/ExCuSe)
add_subdirectory(bindings/python/src/pupil/PuRe)
add_subdirectory(bindings/python/src/pupil/PuReST)
add_subdirectory(bindings/python/src/pupil/PupilLabs2D)
add_subdirectory(bindings/python/src/pupil/Simple)
add_subdirectory(bindings/python/src/pupil/Starburst)
add_subdirectory(bindings/python/src/pupil/Swirski2D)
add_subdirectory(bindings/python/src/glint/Simple)
