nanobind_add_module(Starburst_core_nb
  STABLE_ABI
  NB_STATIC
  src/auto_seed.cpp
  src/contour_detection.cpp
  src/core.cpp
  src/corneal_reflection.cpp
  src/ransac_ellipse.cpp
)
set_target_properties(Starburst_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(Starburst_core_nb
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Starburst_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm_common
)

nanobind_add_stub(Starburst_core_stub
  MODULE _core
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:Starburst_core_nb>
  DEPENDS Starburst_core_nb
)

install(TARGETS Starburst_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/Starburst
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  DESTINATION cheshm/pupil_detectors/Starburst
)
