# Three self-contained C kernels loaded at runtime via ctypes (no Python.h,
# no Cython). Built as plain shared libraries with no "lib" prefix so the
# ctypes loader in pupil_guided_contour.py etc. can find them by their
# source-file stem.

set(LAVAN_C_KERNELS
  active_contour_core
  integro_differential_operator_core
  pupil_guided_contour_core
)

foreach(kernel ${LAVAN_C_KERNELS})
  add_library(${kernel} SHARED ${kernel}.c)
  set_target_properties(${kernel} PROPERTIES
    PREFIX ""
    OUTPUT_NAME ${kernel}
    POSITION_INDEPENDENT_CODE ON
  )
  if(NOT MSVC)
    # libm for sin/cos/sqrt/...; not needed under MSVC's CRT.
    target_link_libraries(${kernel} PRIVATE m)
  endif()
  install(TARGETS ${kernel}
    LIBRARY DESTINATION lavan/boundary
    RUNTIME DESTINATION lavan/boundary
  )
endforeach()
