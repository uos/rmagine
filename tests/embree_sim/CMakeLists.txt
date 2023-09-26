
# 1. SPHERICAL
add_executable(rmagine_tests_embree_simulation_spherical embree_simulation_spherical.cpp)
target_link_libraries(rmagine_tests_embree_simulation_spherical
    rmagine::embree
)

add_test(NAME embree_simulation_spherical COMMAND rmagine_tests_embree_simulation_spherical)

# 2. PINHOLE
add_executable(rmagine_tests_embree_simulation_pinhole embree_simulation_pinhole.cpp)
target_link_libraries(rmagine_tests_embree_simulation_pinhole
    rmagine::embree
)

add_test(NAME embree_simulation_pinhole COMMAND rmagine_tests_embree_simulation_pinhole)

# 3. O1DN
add_executable(rmagine_tests_embree_simulation_o1dn embree_simulation_o1dn.cpp)
target_link_libraries(rmagine_tests_embree_simulation_o1dn
    rmagine::embree
)

add_test(NAME embree_simulation_o1dn COMMAND rmagine_tests_embree_simulation_o1dn)


# 4. ONDN
add_executable(rmagine_tests_embree_simulation_ondn embree_simulation_ondn.cpp)
target_link_libraries(rmagine_tests_embree_simulation_ondn
    rmagine::embree
)

add_test(NAME embree_simulation_ondn COMMAND rmagine_tests_embree_simulation_ondn)

# 5. CLOSEST POINT
add_executable(rmagine_tests_embree_closest_point embree_closest_point.cpp)
target_link_libraries(rmagine_tests_embree_closest_point
    rmagine::embree
)

add_test(NAME embree_closest_point COMMAND rmagine_tests_embree_closest_point)