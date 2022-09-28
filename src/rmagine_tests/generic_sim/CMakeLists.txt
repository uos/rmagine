add_executable(rmagine_tests_generic_sim Main.cpp)

target_link_libraries(rmagine_tests_generic_sim
    rmagine_embree
    rmagine_optix
)