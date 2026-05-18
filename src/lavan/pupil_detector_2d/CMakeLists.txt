# Vendored from pupil-labs/pupil-detectors (LGPL v3). Rewritten for
# scikit-build-core: upstream's pyx -> cpp generation used
# add_cython_target() from scikit-build classic; we generate the .cpp
# manually via `python -m cython --cplus`, then build the Python
# extension with python_add_library(... MODULE ... WITH_SOABI).
#
# Project-wide settings (C++17, flags, find_package Python/Eigen3/OpenCV)
# live in lavan/CMakeLists.txt; this file inherits them.

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/detector_base.cpp
  COMMAND Python::Interpreter -m cython --cplus
          "${CMAKE_CURRENT_SOURCE_DIR}/detector_base.pyx"
          --output-file "${CMAKE_CURRENT_BINARY_DIR}/detector_base.cpp"
  DEPENDS
    detector_base.pyx
    detector_base.pxd
    c_types_wrapper.pxd
    coarse_pupil.pxd
  COMMENT "Cythonizing detector_base.pyx -> detector_base.cpp"
  VERBATIM
)

python_add_library(detector_base MODULE
  ${CMAKE_CURRENT_BINARY_DIR}/detector_base.cpp
  WITH_SOABI
)

install(TARGETS detector_base DESTINATION lavan/pupil_detector_2d)

add_subdirectory(detector_2d)
