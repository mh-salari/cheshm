nanobind_add_module(Starburst_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/Starburst/auto_seed.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/Starburst/contour_detection.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/Starburst/corneal_reflection.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/Starburst/ransac_ellipse.cpp
)
set_target_properties(Starburst_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(Starburst_core_nb
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Starburst_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm::cpp
)

nanobind_add_stub(Starburst_core_stub
  MODULE _core
  OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/Starburst/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:Starburst_core_nb>
  DEPENDS Starburst_core_nb
)

install(TARGETS Starburst_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/Starburst
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/Starburst/_core.pyi
  DESTINATION cheshm/pupil_detectors/Starburst
)
