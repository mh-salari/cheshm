# ElSe pupil detector — nanobind extension `_core`. The Python entry
# point in `core.py` loads it via `from . import _core`. Algorithm body
# lives in `src/else.cpp`; `src/core.cpp` is the binding.

nanobind_add_module(ElSe_core_nb
  STABLE_ABI
  NB_STATIC
  src/core.cpp
  src/else.cpp
)
set_target_properties(ElSe_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(ElSe_core_nb
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ElSe_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

nanobind_add_stub(ElSe_core_stub
  MODULE _core
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:ElSe_core_nb>
  DEPENDS ElSe_core_nb
)

install(TARGETS ElSe_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/ElSe
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  DESTINATION cheshm/pupil_detectors/ElSe
)
