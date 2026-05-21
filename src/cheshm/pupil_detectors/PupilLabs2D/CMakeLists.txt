find_package(Eigen3 CONFIG REQUIRED)

nanobind_add_module(PupilLabs2D_core_nb
  STABLE_ABI
  NB_STATIC
  src/core.cpp
  src/pupil_labs_2d.cpp
  cpp/singleeyefitter/detectorUtils.cpp
)
set_target_properties(PupilLabs2D_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(PupilLabs2D_core_nb
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/shared/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/singleeyefitter
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(PupilLabs2D_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    cheshm_common
)

nanobind_add_stub(PupilLabs2D_core_stub
  MODULE _core
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:PupilLabs2D_core_nb>
  DEPENDS PupilLabs2D_core_nb
)

install(TARGETS PupilLabs2D_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/PupilLabs2D
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  DESTINATION cheshm/pupil_detectors/PupilLabs2D
)
