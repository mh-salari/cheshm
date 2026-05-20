# ExCuSe pupil detector — nanobind extension `_core`. core.py loads it
# directly with `from . import _core`. Algorithm body lives in
# `src/excuse.cpp`; `src/core.cpp` is the binding.

nanobind_add_module(ExCuSe_core_nb
  STABLE_ABI
  NB_STATIC
  src/core.cpp
  src/excuse.cpp
)
set_target_properties(ExCuSe_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(ExCuSe_core_nb
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ExCuSe_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

install(TARGETS ExCuSe_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/ExCuSe
)
