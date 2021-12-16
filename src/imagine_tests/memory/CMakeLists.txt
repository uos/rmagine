add_executable(imagine_tests_memory Main.cpp)

target_link_libraries(imagine_tests_memory
    imagine
    imagine_cuda
)