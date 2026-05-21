nanobind_add_module(active_contour_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
  ${CMAKE_SOURCE_DIR}/src/limbus/Daugman/active_contour/contour.cpp
  ${CMAKE_SOURCE_DIR}/src/limbus/Daugman/active_contour/radial_search.cpp
)
set_target_properties(active_contour_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(active_contour_core_nb
  PRIVATE
    ${CMAKE_SOURCE_DIR}/bindings/python/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(active_contour_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm::cpp
)

nanobind_add_stub(active_contour_core_stub
  MODULE _core
  OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/limbus_detectors/daugman/active_contour/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:active_contour_core_nb>
  DEPENDS active_contour_core_nb
)

install(TARGETS active_contour_core_nb
  LIBRARY DESTINATION cheshm/limbus_detectors/daugman/active_contour
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/limbus_detectors/daugman/active_contour/_core.pyi
  DESTINATION cheshm/limbus_detectors/daugman/active_contour
)
