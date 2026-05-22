find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

nanobind_add_module(viz_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
  ${CMAKE_SOURCE_DIR}/src/viz/render.cpp
)
set_target_properties(viz_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(viz_core_nb
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(viz_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm::cpp
)

nanobind_add_stub(viz_core_stub
  MODULE _core
  OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/viz/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:viz_core_nb>
  DEPENDS viz_core_nb
)

install(TARGETS viz_core_nb
  LIBRARY DESTINATION cheshm/viz
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/viz/_core.pyi
  DESTINATION cheshm/viz
)
