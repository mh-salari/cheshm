nanobind_add_module(pupil_guided_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
  ${CMAKE_SOURCE_DIR}/src/limbus/Daugman/pupil_guided/contour.cpp
  ${CMAKE_SOURCE_DIR}/src/limbus/Daugman/pupil_guided/radial_search.cpp
)
set_target_properties(pupil_guided_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(pupil_guided_core_nb
  PRIVATE
    ${CMAKE_SOURCE_DIR}/bindings/python/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(pupil_guided_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm::cpp
)

if(CHESHM_BUILD_STUBS)
  nanobind_add_stub(pupil_guided_core_stub
    ALL
    MODULE _core
    OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/limbus_detectors/daugman/pupil_guided/_core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:pupil_guided_core_nb>
    DEPENDS pupil_guided_core_nb
  )
endif()

install(TARGETS pupil_guided_core_nb
  LIBRARY DESTINATION cheshm/limbus_detectors/daugman/pupil_guided
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/limbus_detectors/daugman/pupil_guided/_core.pyi
  DESTINATION cheshm/limbus_detectors/daugman/pupil_guided
)
