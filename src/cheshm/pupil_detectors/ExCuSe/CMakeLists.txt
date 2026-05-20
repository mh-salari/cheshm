# ExCuSe pupil detector — compiles a "core" shared library that
# core.py loads via ctypes. Edge-based detection with adaptive
# thresholding, ray-based contour collection, and ellipse fitting.

add_library(ExCuSe_core SHARED
  src/core.cpp
  src/excuse.cpp
)

target_include_directories(ExCuSe_core
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ExCuSe_core
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

set_target_properties(ExCuSe_core PROPERTIES
  PREFIX ""
  OUTPUT_NAME "core"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

install(TARGETS ExCuSe_core
  LIBRARY DESTINATION cheshm/pupil_detectors/ExCuSe
  RUNTIME DESTINATION cheshm/pupil_detectors/ExCuSe
)
