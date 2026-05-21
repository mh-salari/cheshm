nanobind_add_module(PuReST_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/PuReST/purest.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/PuRe/pure.cpp
)
set_target_properties(PuReST_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(PuReST_core_nb
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(PuReST_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm::cpp
)

nanobind_add_stub(PuReST_core_stub
  MODULE _core
  OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/PuReST/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:PuReST_core_nb>
  DEPENDS PuReST_core_nb
)

install(TARGETS PuReST_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/PuReST
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/PuReST/_core.pyi
  DESTINATION cheshm/pupil_detectors/PuReST
)
