nanobind_add_module(ExCuSe_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/ExCuSe/excuse.cpp
)
set_target_properties(ExCuSe_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(ExCuSe_core_nb
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ExCuSe_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    cheshm::cpp
)

if(CHESHM_BUILD_STUBS)
  nanobind_add_stub(ExCuSe_core_stub
    ALL
    MODULE _core
    OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/ExCuSe/_core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:ExCuSe_core_nb>
    DEPENDS ExCuSe_core_nb
  )
endif()

install(TARGETS ExCuSe_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/ExCuSe
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/ExCuSe/_core.pyi
  DESTINATION cheshm/pupil_detectors/ExCuSe
)
