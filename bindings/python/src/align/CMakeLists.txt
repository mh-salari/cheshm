find_package(Threads REQUIRED)

nanobind_add_module(align_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
  ${CMAKE_SOURCE_DIR}/src/align/rigid.cpp
)
set_target_properties(align_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(align_core_nb
  PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(align_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    cheshm::cpp
)

nanobind_add_stub(align_core_stub
  MODULE _core
  OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/align/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:align_core_nb>
  DEPENDS align_core_nb
)

install(TARGETS align_core_nb
  LIBRARY DESTINATION cheshm/align
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/align/_core.pyi
  DESTINATION cheshm/align
)
