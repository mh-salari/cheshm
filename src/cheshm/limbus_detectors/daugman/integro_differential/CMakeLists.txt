nanobind_add_module(integro_differential_core_nb
  STABLE_ABI
  NB_STATIC
  src/core.cpp
  src/operator.cpp
)
set_target_properties(integro_differential_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(integro_differential_core_nb
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(integro_differential_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

nanobind_add_stub(integro_differential_core_stub
  MODULE _core
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:integro_differential_core_nb>
  DEPENDS integro_differential_core_nb
)

install(TARGETS integro_differential_core_nb
  LIBRARY DESTINATION cheshm/limbus_detectors/daugman/integro_differential
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  DESTINATION cheshm/limbus_detectors/daugman/integro_differential
)
