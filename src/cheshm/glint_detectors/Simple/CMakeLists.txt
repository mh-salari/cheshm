# Simple glint detector — compiles a "core" shared library that
# core.py loads via ctypes. Threshold + pupil-centred search disk +
# optional half-plane filter + shape-quality walk + split-widest fallback;
# five glint-center methods including a periodic interpolating cubic
# spline through the convex hull.

add_library(Simple_glint_core SHARED
  src/core.cpp
)

target_include_directories(Simple_glint_core
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Simple_glint_core
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

set_target_properties(Simple_glint_core PROPERTIES
  PREFIX ""
  OUTPUT_NAME "core"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

install(TARGETS Simple_glint_core
  LIBRARY DESTINATION cheshm/glint_detectors/Simple
  RUNTIME DESTINATION cheshm/glint_detectors/Simple
)
