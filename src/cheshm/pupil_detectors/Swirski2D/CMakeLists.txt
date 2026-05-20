# Swirski 2D pupil detector — compiles a "core" shared library that
# core.py loads via ctypes. The detector pulls in OpenCV (linked
# privately) and exposes a single extern "C" entry point.

add_library(Swirski2D_core SHARED
  src/core.cpp
  src/cvx.cpp
  src/pupil_tracker.cpp
  src/utils.cpp
)

target_include_directories(Swirski2D_core
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
)

# poolSTL needs std::thread / std::mutex; explicit on Linux toolchains.
find_package(Threads REQUIRED)

target_link_libraries(Swirski2D_core
  PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    cheshm_common
)

set_target_properties(Swirski2D_core PROPERTIES
  PREFIX ""
  OUTPUT_NAME "core"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

install(TARGETS Swirski2D_core
  LIBRARY DESTINATION cheshm/pupil_detectors/Swirski2D
  RUNTIME DESTINATION cheshm/pupil_detectors/Swirski2D
)
