
add_library(my_mesh_changer mesh_changer.cu)

target_link_libraries(my_mesh_changer
    rmagine-core
)

add_executable(rmagine_examples_optix_scene Main.cpp)

target_link_libraries(rmagine_examples_optix_scene
    rmagine-core
    rmagine-optix
    my_mesh_changer
)