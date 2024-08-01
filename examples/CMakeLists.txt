add_executable(01ParticleArrayCreation particle_array_creation.cpp)
target_link_libraries(01ParticleArrayCreation PRIVATE cabanadem)

add_executable(TwoBlocksColliding two_block_colliding.cpp)
target_link_libraries(TwoBlocksColliding PRIVATE cabanadem)

install(TARGETS 01ParticleArrayCreation TwoBlocksColliding DESTINATION ${CMAKE_INSTALL_BINDIR})



add_definitions(-w)
