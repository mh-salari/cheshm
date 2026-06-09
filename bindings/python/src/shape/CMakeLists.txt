nanobind_add_module(shape_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
)
set_target_properties(shape_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(shape_core_nb
  PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(shape_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm::cpp
)

if(CHESHM_BUILD_STUBS)
  nanobind_add_stub(shape_core_stub
    ALL
    MODULE _core
    OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/shape/_core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:shape_core_nb>
    DEPENDS shape_core_nb
  )
endif()

install(TARGETS shape_core_nb
  LIBRARY DESTINATION cheshm/shape
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/shape/_core.pyi
  DESTINATION cheshm/shape
)
