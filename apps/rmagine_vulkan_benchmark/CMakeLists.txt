if(Vulkan_FOUND)

add_executable(rmagine_vulkan_benchmark
    Main.cpp
)

target_link_libraries(rmagine_vulkan_benchmark
    rmagine::core
    rmagine::vulkan
)

target_compile_definitions(rmagine_vulkan_benchmark PUBLIC VDEBUG FORCE_SHADER_RECOMPILE)

##### INSTALL
install(TARGETS rmagine_vulkan_benchmark
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT vulkan
)



#dirs for shaders:

#create dirs for shaders
set(SHADER_BINARY_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders")
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

#copy shadersources to bin dir
set(SHADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/rmagine_vulkan/shaders")
set(SHADER_COPY_SOURCE_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaderSources")
file(MAKE_DIRECTORY ${SHADER_COPY_SOURCE_DIR})
file(GLOB SHADER_SOURCES 
    "${SHADER_SOURCE_DIR}/*.rgen"
    "${SHADER_SOURCE_DIR}/*.rchit"
    "${SHADER_SOURCE_DIR}/*.rmiss"
    "${SHADER_SOURCE_DIR}/*.rcall"
    "${SHADER_SOURCE_DIR}/*.glsl"
)
file(COPY ${SHADER_SOURCES} DESTINATION ${SHADER_COPY_SOURCE_DIR})

foreach (SHADER_FILE IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    file(MAKE_DIRECTORY "${SHADER_BINARY_DIR}/${SHADER_NAME}")
endforeach ()


file(GLOB SENSOR_TYPE_DEPENDENT_SHADER_SOURCES 
    "${SHADER_SOURCE_DIR}/*.rgen"
)
file(GLOB RESULT_TYPE_DEPENDENT_SHADER_SOURCES 
    "${SHADER_SOURCE_DIR}/*.rchit"
    "${SHADER_SOURCE_DIR}/*.rmiss"
)

endif(Vulkan_FOUND)