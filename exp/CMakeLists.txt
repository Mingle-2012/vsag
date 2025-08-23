file(GLOB ALL_CPP_FILES "*.cpp")

foreach(SOURCE_FILE ${ALL_CPP_FILES})
    get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)

    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})

    target_link_libraries(${EXECUTABLE_NAME} PRIVATE
            vsag
            simd
    )
    target_include_directories (${EXECUTABLE_NAME} PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include
            ${HDF5_INCLUDE_DIRS})

    add_dependencies (${EXECUTABLE_NAME} spdlog)

endforeach()