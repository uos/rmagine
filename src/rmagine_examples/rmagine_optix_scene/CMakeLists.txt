
add_library(my_mesh_changer mesh_changer.cu)

add_executable(rmagine_examples_optix_scene Main.cpp)

target_link_libraries(rmagine_examples_optix_scene
    rmagine
    rmagine_optix
    my_mesh_changer
)