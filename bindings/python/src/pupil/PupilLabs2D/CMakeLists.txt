find_package(Eigen3 CONFIG REQUIRED)

nanobind_add_module(PupilLabs2D_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/PupilLabs2D/pupil_labs_2d.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/PupilLabs2D/cpp/singleeyefitter/detectorUtils.cpp
)
set_target_properties(PupilLabs2D_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(PupilLabs2D_core_nb
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pupil/PupilLabs2D/cpp
    ${CMAKE_SOURCE_DIR}/src/pupil/PupilLabs2D/cpp/shared/include
    ${CMAKE_SOURCE_DIR}/src/pupil/PupilLabs2D/cpp/singleeyefitter
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(PupilLabs2D_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    cheshm::cpp
)

# Upstream singleeyefitter math headers use M_PI, which MSVC does not
# expose unless _USE_MATH_DEFINES is set before <cmath> / <math.h>.
target_compile_definitions(PupilLabs2D_core_nb PRIVATE _USE_MATH_DEFINES)

if(CHESHM_BUILD_STUBS)
  nanobind_add_stub(PupilLabs2D_core_stub
    ALL
    MODULE _core
    OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/PupilLabs2D/_core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:PupilLabs2D_core_nb>
    DEPENDS PupilLabs2D_core_nb
  )
endif()

install(TARGETS PupilLabs2D_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/PupilLabs2D
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/PupilLabs2D/_core.pyi
  DESTINATION cheshm/pupil_detectors/PupilLabs2D
)
