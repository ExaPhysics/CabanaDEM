add_executable(01SphereSphereCollision 01SphereSphereCollision.cpp)
target_link_libraries(01SphereSphereCollision PRIVATE cabanadem)

install(TARGETS 01SphereSphereCollision DESTINATION ${CMAKE_INSTALL_BINDIR})


add_definitions(-w)
