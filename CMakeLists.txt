cmake_minimum_required(VERSION 3.8)

project(FFTBenchmarks)

# Update CMAKE_MODULE_PATH with local CMake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(MPI)
find_package(FFTW REQUIRED)
find_package(Boost REQUIRED
    COMPONENTS
    program_options
    )

# FFTW transformation over a 2D plane, back and forth
add_executable(
    fftw_2d_plane
    fftw_2d_plane.cpp
)

target_include_directories(fftw_2d_plane PUBLIC
    ${FFTW_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_PATH}
    ${Boost_INCLUDE_DIRS}
    )

target_link_libraries(fftw_2d_plane PUBLIC
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${FFTW_MPI_LIBRARY} ${FFTW_LIBRARY}
    ${MPI_CXX_LIBRARIES}
    )

