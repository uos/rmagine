
add_executable(rmagine_examples_embree_scene Main.cpp)

target_link_libraries(rmagine_examples_embree_scene
    rmagine
    rmagine_embree
)