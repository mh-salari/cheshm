# Starburst pupil detector — compiles a "core" shared library that
# core.py loads via ctypes. The detector pulls in OpenCV (linked
# privately) and exposes a single extern "C" entry point.

add_library(Starburst_core SHARED
  src/core.cpp
  src/contour_detection.cpp
  src/corneal_reflection.cpp
  src/ransac_ellipse.cpp
  src/svd.cpp
)

target_include_directories(Starburst_core
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Starburst_core
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

set_target_properties(Starburst_core PROPERTIES
  PREFIX ""
  OUTPUT_NAME "core"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

install(TARGETS Starburst_core
  LIBRARY DESTINATION cheshm/pupil_detectors/Starburst
  RUNTIME DESTINATION cheshm/pupil_detectors/Starburst
)
