file(GLOB_RECURSE ALL_HPP "*.hpp")
file(GLOB ALL_CPP_FILES "*.cpp")

find_package(OpenMP REQUIRED)

foreach(SOURCE_FILE ${ALL_CPP_FILES})
    get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)

    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE} ${ALL_HPP})

    target_link_libraries(${EXECUTABLE_NAME} PRIVATE
            vsag
            simd
            OpenMP::OpenMP_CXX
    )
    target_include_directories (${EXECUTABLE_NAME} PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include
            ${HDF5_INCLUDE_DIRS})

    add_dependencies (${EXECUTABLE_NAME} spdlog)

endforeach()