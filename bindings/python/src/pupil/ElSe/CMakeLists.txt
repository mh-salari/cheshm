nanobind_add_module(ElSe_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/ElSe/else.cpp
)
set_target_properties(ElSe_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(ElSe_core_nb
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ElSe_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm::cpp
)

if(CHESHM_BUILD_STUBS)
  nanobind_add_stub(ElSe_core_stub
    ALL
    MODULE _core
    OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/ElSe/_core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:ElSe_core_nb>
    DEPENDS ElSe_core_nb
  )
endif()

install(TARGETS ElSe_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/ElSe
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/ElSe/_core.pyi
  DESTINATION cheshm/pupil_detectors/ElSe
)
