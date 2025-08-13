if(Vulkan_FOUND)

add_executable(rmagine_vulkan_general_test
    Main.cpp
)

target_link_libraries(rmagine_vulkan_general_test
    rmagine::core
    rmagine::vulkan
)

target_compile_definitions(rmagine_vulkan_general_test PUBLIC VDEBUG FORCE_SHADER_RECOMPILE)

##### INSTALL
install(TARGETS rmagine_vulkan_general_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT vulkan
)



#compile shaders:

#create dirs for shaders
set(SHADER_BINARY_DIR "${CMAKE_INSTALL_BINDIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

#copy shadersources to bin dir
set(SHADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/rmagine_vulkan/shaders")
set(SHADER_COPY_SOURCE_DIR "${CMAKE_INSTALL_BINDIR}/shaderSources")
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


#Initialize an empty list to store .spv output
set(SHADER_OUTPUTS)


#Loop through each result type dependent shader file and compile with RANGES
foreach (SHADER_FILE IN LISTS RESULT_TYPE_DEPENDENT_SHADER_SOURCES)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    set(OUTPUT_PATH "${SHADER_BINARY_DIR}/${SHADER_NAME}/RANGES.spv")
    list(APPEND SHADER_OUTPUTS ${OUTPUT_PATH})

    add_custom_command(
        OUTPUT "${OUTPUT_PATH}"
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} --target-env vulkan1.3 --define-macro RANGES=def -o ${OUTPUT_PATH} ${SHADER_FILE}
        DEPENDS ${SHADER_FILE}
        COMMENT "Compiling : ${SHADER_NAME}"
    )
endforeach ()


#list of defines used in universal ray generation shader
set(SENSOR_DEFINES
    "SPHERE"
    "PINHOLE"
    "O1DN"
    "ONDN"
)

#Loop through each sensor type dependent shader file and compile for each sensor type
foreach (SHADER_FILE IN LISTS SENSOR_TYPE_DEPENDENT_SHADER_SOURCES)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)

    #Loop through the different sensor types
    foreach(RGEN_DEFINE IN LISTS RGEN_OUTPUT_NAMES SENSOR_DEFINES)
        set(OUTPUT_PATH "${SHADER_BINARY_DIR}/${SHADER_NAME}/${RGEN_DEFINE}.spv")
        list(APPEND SHADER_OUTPUTS ${OUTPUT_PATH})
        
        add_custom_command(
            OUTPUT "${OUTPUT_PATH}"
            COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} --target-env vulkan1.3 --define-macro ${RGEN_DEFINE}=def -o ${OUTPUT_PATH} ${SHADER_FILE}
            DEPENDS ${SHADER_FILE}
            COMMENT "Compiling : ${SHADER_NAME} (${RGEN_DEFINE})"
        )
    endforeach()
endforeach ()


message(STATUS "Shader outputs : ${SHADER_OUTPUTS}")

endif(Vulkan_FOUND)