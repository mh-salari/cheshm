# Starburst pupil detector — compiles a "core" shared library that
# core.py loads via ctypes. The detector pulls in OpenCV (linked
# privately) and exposes a single extern "C" entry point.

add_library(starburst_core SHARED
  src/core.cpp
  src/contour_detection.cpp
  src/corneal_reflection.cpp
  src/ransac_ellipse.cpp
  src/svd.cpp
)

target_include_directories(starburst_core
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(starburst_core
  PRIVATE
    ${OpenCV_LIBS}
)

set_target_properties(starburst_core PROPERTIES
  PREFIX ""
  OUTPUT_NAME "core"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

install(TARGETS starburst_core
  LIBRARY DESTINATION lavan/pupil_detectors/starburst
  RUNTIME DESTINATION lavan/pupil_detectors/starburst
)
