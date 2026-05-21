nanobind_add_module(Simple_core_nb
  STABLE_ABI
  NB_STATIC
  src/core.cpp
)
set_target_properties(Simple_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(Simple_core_nb
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Simple_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

nanobind_add_stub(Simple_core_stub
  MODULE _core
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:Simple_core_nb>
  DEPENDS Simple_core_nb
)

install(TARGETS Simple_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/Simple
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  DESTINATION cheshm/pupil_detectors/Simple
)
