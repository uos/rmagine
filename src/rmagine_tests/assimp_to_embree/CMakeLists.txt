add_executable(rmagine_tests_assimp_to_embree Main.cpp)

target_link_libraries(rmagine_tests_assimp_to_embree
    rmagine
    rmagine_embree
)