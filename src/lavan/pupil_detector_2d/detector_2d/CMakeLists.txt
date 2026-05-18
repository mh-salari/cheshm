# Vendored from pupil-labs/pupil-detectors detector_2d/ (LGPL v3).
# Rewritten for scikit-build-core; project-wide settings come from
# lavan/CMakeLists.txt (C++17, find_package Python/NumPy/Eigen3/OpenCV).

cmake_policy(SET CMP0076 NEW)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/detector_2d.cpp
  COMMAND Python::Interpreter -m cython --cplus
          "${CMAKE_CURRENT_SOURCE_DIR}/detector_2d.pyx"
          --output-file "${CMAKE_CURRENT_BINARY_DIR}/detector_2d.cpp"
  DEPENDS
    detector_2d.pyx
    detector_2d.pxd
    detect_2d.hpp
  COMMENT "Cythonizing detector_2d.pyx -> detector_2d.cpp"
  VERBATIM
)

python_add_library(detector_2d MODULE
  ${CMAKE_CURRENT_BINARY_DIR}/detector_2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/singleeyefitter/detectorUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/singleeyefitter/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/singleeyefitter/ImageProcessing/cvx.cpp
  WITH_SOABI
)

target_include_directories(detector_2d PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Python_NumPy_INCLUDE_DIRS}

  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/shared/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/singleeyefitter
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/singleeyefitter/ImageProcessing
)

target_link_libraries(detector_2d PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)

install(TARGETS detector_2d DESTINATION lavan/pupil_detector_2d/detector_2d)
