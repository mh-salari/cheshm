set(CHESHM_DAUGMAN_DETECTORS
  active_contour
  integro_differential
  pupil_guided
)

foreach(detector ${CHESHM_DAUGMAN_DETECTORS})
  set(target "${detector}_core")
  add_library(${target} SHARED ${detector}/core.c)
  set_target_properties(${target} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "core"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${detector}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${detector}"
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )
  if(NOT MSVC)
    target_link_libraries(${target} PRIVATE m)
  endif()
  install(TARGETS ${target}
    LIBRARY DESTINATION cheshm/limbus_detectors/daugman/${detector}
    RUNTIME DESTINATION cheshm/limbus_detectors/daugman/${detector}
  )
endforeach()
