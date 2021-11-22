

# Capsule PTX-Strings in files


foreach(OPTIX_KERNEL_FILE ${OPTIX_KERNEL_FILES})
    # Get Name of Kernel
    get_filename_component(OPTIX_KERNEL_NAME ${OPTIX_KERNEL_FILE} NAME_WLE)
    # Read Compiled Kernel to String
    file(READ "${IMAGINE_OPTIX_PTX_DIR}/cuda_compile_ptx_1_generated_${OPTIX_KERNEL_NAME}.cu.ptx" INCLUDE_STRING)
    # Write to static readable file e.g. R("")
    configure_file(${IMAGINE_SOURCE_DIR}/cmake/FileToString.h.in "include/kernels/${OPTIX_KERNEL_NAME}String.h")
endforeach()



