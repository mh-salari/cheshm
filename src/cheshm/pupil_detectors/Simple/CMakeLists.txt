# Simple pupil detector — compiles a "core" shared library that
# core.py loads via ctypes. Threshold + contour-find + shape-quality
# walk + ellipse fit; four pupil-center methods including a periodic
# interpolating cubic spline through the convex hull.

add_library(Simple_core SHARED
  src/core.cpp
)

target_include_directories(Simple_core
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Simple_core
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

set_target_properties(Simple_core PROPERTIES
  PREFIX ""
  OUTPUT_NAME "core"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

install(TARGETS Simple_core
  LIBRARY DESTINATION cheshm/pupil_detectors/Simple
  RUNTIME DESTINATION cheshm/pupil_detectors/Simple
)
