# Swirski 2D pupil detector — nanobind extension `_core`. core.py loads
# it directly with `from . import _core`.

# poolSTL needs std::thread / std::mutex; explicit on Linux toolchains.
find_package(Threads REQUIRED)

nanobind_add_module(Swirski2D_core_nb
  STABLE_ABI
  NB_STATIC
  src/core.cpp
  src/cvx.cpp
  src/pupil_tracker.cpp
  src/utils.cpp
)
set_target_properties(Swirski2D_core_nb PROPERTIES OUTPUT_NAME "_core")

target_include_directories(Swirski2D_core_nb
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Swirski2D_core_nb
  PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    cheshm_common
)

nanobind_add_stub(Swirski2D_core_stub
  MODULE _core
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:Swirski2D_core_nb>
  DEPENDS Swirski2D_core_nb
)

install(TARGETS Swirski2D_core_nb
  LIBRARY DESTINATION cheshm/pupil_detectors/Swirski2D
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/_core.pyi
  DESTINATION cheshm/pupil_detectors/Swirski2D
)
