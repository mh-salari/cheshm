cmake_minimum_required(VERSION 3.25)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
  set(CMAKE_C_FLAGS_RELEASE "-O3 ${CMAKE_C_FLAGS_RELEASE}")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(src/lavan/limbus_detectors/daugman)
