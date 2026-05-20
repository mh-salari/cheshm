# Simple glint detector — nanobind extension `_core`. core.py loads it
# directly with `from . import _core` and provides the public Python API.

nanobind_add_module(Simple_glint_core_nb
  STABLE_ABI
  NB_STATIC
  src/core.cpp
)
set_target_properties(Simple_glint_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(Simple_glint_core_nb
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Simple_glint_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

install(TARGETS Simple_glint_core_nb
  LIBRARY DESTINATION cheshm/glint_detectors/Simple
)
