nanobind_add_module(enhance_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
  ${CMAKE_SOURCE_DIR}/src/enhance/enhance.cpp
)
set_target_properties(enhance_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(enhance_core_nb
  PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(enhance_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm::cpp
)

if(CHESHM_BUILD_STUBS)
  nanobind_add_stub(enhance_core_stub
    ALL
    MODULE _core
    OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/enhance/_core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:enhance_core_nb>
    DEPENDS enhance_core_nb
  )
endif()

install(TARGETS enhance_core_nb
  LIBRARY DESTINATION cheshm/enhance
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/enhance/_core.pyi
  DESTINATION cheshm/enhance
)
