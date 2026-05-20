# Swirski 2D pupil detector — compiles a "core" shared library that
# core.py loads via ctypes. The detector pulls in OpenCV (linked
# privately) and exposes a single extern "C" entry point.

add_library(swirski_2d_core SHARED
  src/core.cpp
  src/cvx.cpp
  src/pupil_tracker.cpp
  src/utils.cpp
)

target_include_directories(swirski_2d_core
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
)

# poolSTL needs std::thread / std::mutex; explicit on Linux toolchains.
find_package(Threads REQUIRED)

target_link_libraries(swirski_2d_core
  PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

set_target_properties(swirski_2d_core PROPERTIES
  PREFIX ""
  OUTPUT_NAME "core"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

install(TARGETS swirski_2d_core
  LIBRARY DESTINATION lavan/pupil_detectors/swirski_2d
  RUNTIME DESTINATION lavan/pupil_detectors/swirski_2d
)
