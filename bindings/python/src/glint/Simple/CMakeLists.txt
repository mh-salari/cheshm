nanobind_add_module(Simple_glint_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
)
set_target_properties(Simple_glint_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(Simple_glint_core_nb
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Simple_glint_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm::cpp
)

if(CHESHM_BUILD_STUBS)
  nanobind_add_stub(Simple_glint_core_stub
    ALL
    MODULE _core
    OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/glint_detectors/Simple/_core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:Simple_glint_core_nb>
    DEPENDS Simple_glint_core_nb
  )
endif()

install(TARGETS Simple_glint_core_nb
  LIBRARY DESTINATION cheshm/glint_detectors/Simple
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/glint_detectors/Simple/_core.pyi
  DESTINATION cheshm/glint_detectors/Simple
)
