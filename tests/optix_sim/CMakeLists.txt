
# 1. SPHERICAL

add_executable(rmagine_tests_optix_simulation_spherical optix_simulation_spherical.cpp)
target_link_libraries(rmagine_tests_optix_simulation_spherical
    rmagine::optix
)

add_test(NAME optix_simulation_spherical COMMAND rmagine_tests_optix_simulation_spherical)

# 2. PINHOLE


add_executable(rmagine_tests_optix_simulation_pinhole optix_simulation_pinhole.cpp)
target_link_libraries(rmagine_tests_optix_simulation_pinhole
    rmagine::optix
)

add_test(NAME optix_simulation_pinhole COMMAND rmagine_tests_optix_simulation_pinhole)

# 3. O1DN
add_executable(rmagine_tests_optix_simulation_o1dn optix_simulation_o1dn.cpp)
target_link_libraries(rmagine_tests_optix_simulation_o1dn
    rmagine::optix
)

add_test(NAME optix_simulation_o1dn COMMAND rmagine_tests_optix_simulation_o1dn)


# 4. ONDN
add_executable(rmagine_tests_optix_simulation_ondn optix_simulation_ondn.cpp)
target_link_libraries(rmagine_tests_optix_simulation_ondn
    rmagine::optix
)

add_test(NAME optix_simulation_ondn COMMAND rmagine_tests_optix_simulation_ondn)