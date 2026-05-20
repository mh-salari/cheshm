nanobind_add_module(PuRe_core_nb
  STABLE_ABI
  NB_STATIC
  src/core.cpp
  src/pure.cpp
)
set_target_properties(PuRe_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(PuRe_core_nb
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(PuRe_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

nanobind_add_stub(PuRe_core_stub
  MODULE _core
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:PuRe_core_nb>
  DEPENDS PuRe_core_nb
)

install(TARGETS PuRe_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/PuRe
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  DESTINATION cheshm/pupil_detectors/PuRe
)