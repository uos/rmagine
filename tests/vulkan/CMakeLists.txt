
# 1. SPHERICAL
add_executable(rmagine_tests_vulkan_simulation_spherical simulation_spherical.cpp)
target_link_libraries(rmagine_tests_vulkan_simulation_spherical
    rmagine::vulkan
)

add_test(NAME vulkan_simulation_spherical COMMAND rmagine_tests_vulkan_simulation_spherical)


# 2. PINHOLE
add_executable(rmagine_tests_vulkan_simulation_pinhole simulation_pinhole.cpp)
target_link_libraries(rmagine_tests_vulkan_simulation_pinhole
    rmagine::vulkan
)

add_test(NAME vulkan_simulation_pinhole COMMAND rmagine_tests_vulkan_simulation_pinhole)


# 3. O1DN
add_executable(rmagine_tests_vulkan_simulation_o1dn simulation_o1dn.cpp)
target_link_libraries(rmagine_tests_vulkan_simulation_o1dn
    rmagine::vulkan
)

add_test(NAME vulkan_simulation_o1dn COMMAND rmagine_tests_vulkan_simulation_o1dn)


# 4. ONDN
add_executable(rmagine_tests_vulkan_simulation_ondn simulation_ondn.cpp)
target_link_libraries(rmagine_tests_vulkan_simulation_ondn
    rmagine::vulkan
)

add_test(NAME vulkan_simulation_ondn COMMAND rmagine_tests_vulkan_simulation_ondn)
