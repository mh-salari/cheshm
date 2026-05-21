nanobind_add_module(pupil_guided_core_nb
  STABLE_ABI
  NB_STATIC
  src/core.cpp
  src/contour.cpp
  src/radial_search.cpp
)
set_target_properties(pupil_guided_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(pupil_guided_core_nb
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../_common/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(pupil_guided_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

nanobind_add_stub(pupil_guided_core_stub
  MODULE _core
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:pupil_guided_core_nb>
  DEPENDS pupil_guided_core_nb
)

install(TARGETS pupil_guided_core_nb
  LIBRARY DESTINATION cheshm/limbus_detectors/daugman/pupil_guided
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  DESTINATION cheshm/limbus_detectors/daugman/pupil_guided
)
