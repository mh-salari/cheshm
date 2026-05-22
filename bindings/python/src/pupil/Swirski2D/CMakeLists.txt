find_package(Threads REQUIRED)

nanobind_add_module(Swirski2D_core_nb
  STABLE_ABI
  NB_STATIC
  core.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/Swirski2D/cvx.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/Swirski2D/pupil_tracker.cpp
  ${CMAKE_SOURCE_DIR}/src/pupil/Swirski2D/utils.cpp
)
set_target_properties(Swirski2D_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(Swirski2D_core_nb
  PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Swirski2D_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    cheshm::cpp
)

if(CHESHM_BUILD_STUBS)
  nanobind_add_stub(Swirski2D_core_stub
    ALL
    MODULE _core
    OUTPUT ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/Swirski2D/_core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:Swirski2D_core_nb>
    DEPENDS Swirski2D_core_nb
  )
endif()

install(TARGETS Swirski2D_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/Swirski2D
)
install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/cheshm/pupil_detectors/Swirski2D/_core.pyi
  DESTINATION cheshm/pupil_detectors/Swirski2D
)
