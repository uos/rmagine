
# 1. SPHERICAL
add_executable(rmagine_tests_embree_simulation_spherical simulation_spherical.cpp)
target_link_libraries(rmagine_tests_embree_simulation_spherical
    rmagine::embree
)

add_test(NAME embree_simulation_spherical COMMAND rmagine_tests_embree_simulation_spherical)

# 2. PINHOLE
add_executable(rmagine_tests_embree_simulation_pinhole simulation_pinhole.cpp)
target_link_libraries(rmagine_tests_embree_simulation_pinhole
    rmagine::embree
)

add_test(NAME embree_simulation_pinhole COMMAND rmagine_tests_embree_simulation_pinhole)

# 3. O1DN
add_executable(rmagine_tests_embree_simulation_o1dn simulation_o1dn.cpp)
target_link_libraries(rmagine_tests_embree_simulation_o1dn
    rmagine::embree
)

add_test(NAME embree_simulation_o1dn COMMAND rmagine_tests_embree_simulation_o1dn)


# 4. ONDN
add_executable(rmagine_tests_embree_simulation_ondn simulation_ondn.cpp)
target_link_libraries(rmagine_tests_embree_simulation_ondn
    rmagine::embree
)

add_test(NAME embree_simulation_ondn COMMAND rmagine_tests_embree_simulation_ondn)

# 5. CLOSEST POINT
add_executable(rmagine_tests_embree_closest_point closest_point.cpp)
target_link_libraries(rmagine_tests_embree_closest_point
    rmagine::embree
)

add_test(NAME embree_closest_point COMMAND rmagine_tests_embree_closest_point)


# 6. RCC CORRECTION
add_executable(rmagine_tests_embree_correction_rcc correction_rcc.cpp)
target_link_libraries(rmagine_tests_embree_correction_rcc
    rmagine::embree
)

add_test(NAME embree_correction_rcc COMMAND rmagine_tests_embree_correction_rcc)


# 6. MAP CAST
add_executable(rmagine_tests_embree_map_cast map_cast.cpp)
target_link_libraries(rmagine_tests_embree_map_cast
    rmagine::embree
)

add_test(NAME embree_map_cast COMMAND rmagine_tests_embree_map_cast)