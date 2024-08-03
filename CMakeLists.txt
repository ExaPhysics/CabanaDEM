# project settings
cmake_minimum_required(VERSION 3.12)

# C only for Cabana HDF5 workaround.
project(CabanaDEM LANGUAGES CXX C VERSION 0.1.0)

include(GNUInstallDirs)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
find_package(Cabana 0.6.1 REQUIRED COMPONENTS Cabana::Grid Cabana::Core)
if( NOT Cabana_ENABLE_MPI )
  message( FATAL_ERROR "Cabana must be compiled with MPI" )
endif()
find_package(nlohmann_json 3.10.0 QUIET)
if(NOT NLOHMANN_JSON_FOUND)
  include(FetchContent)
  # Using most recent release here
  FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()

# find Clang Format
find_package( CLANG_FORMAT 14 )

# library
add_subdirectory(src)

# examples
add_subdirectory(examples)

##---------------------------------------------------------------------------##
## Clang Format
##---------------------------------------------------------------------------##
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.hpp examples/*.cpp examples/*.hpp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
