cmake_minimum_required(VERSION 3.8)

project(FFTBenchmarks LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Update CMAKE_MODULE_PATH with local CMake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(ENABLE_CUDA False CACHE BOOL "Enable the build of GPU benchmark applications")
if (ENABLE_CUDA)
    enable_language(CUDA)
endif ()


find_package(MPI)
find_package(FFTW REQUIRED)
find_package(AccFFT)
find_package(Boost REQUIRED
    COMPONENTS
    program_options
    )


# Sets include directories and link libraries for an FFTW
# application using MPI. It must be called passing the name
# of the application as the first argument, followed by all
# the source files:
#
#     fftw_mpi_application(fftw_example fftw_example.cpp)
#
function(fftw_mpi_application name)
    add_executable(
        ${name}
        ${ARGN}
    )

    target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_include_directories(${name} SYSTEM PUBLIC
        ${FFTW_INCLUDE_DIR}
        ${MPI_CXX_INCLUDE_PATH}
        ${Boost_INCLUDE_DIRS}
        )

    target_link_libraries(${name} PUBLIC
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${FFTW_MPI_LIBRARY} ${FFTW_LIBRARY}
        ${MPI_CXX_LIBRARIES}
        )
endfunction()

function(accfft_mpi_application name)
    add_executable(
        ${name}
        ${ARGN}
    )

    target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_include_directories(${name} SYSTEM PUBLIC
        ${MPI_CXX_INCLUDE_PATH}
        ${ACCFFT_INCLUDE_DIRS}
        ${FFTW_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        )

    target_link_libraries(${name} PUBLIC
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${ACCFFT_CPU_LIBRARY}
        ${ACCFFT_CPU_FFTW_LIBRARIES}
        ${MPI_CXX_LIBRARIES}
        )
endfunction()

function(accfft_mpi_cuda_application name)
    add_executable(
        ${name}
        ${ARGN}
    )

    target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_include_directories(${name} SYSTEM PUBLIC
        ${MPI_CXX_INCLUDE_PATH}
        ${ACCFFT_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        )

    target_link_libraries(${name} PUBLIC
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${ACCFFT_GPU_LIBRARY}
        ${ACCFFT_CPU_FFTW_LIBRARIES}
        ${MPI_CXX_LIBRARIES}
        -lcufft -lcublas
        )
endfunction()

# FFTW
fftw_mpi_application(fftw_2d_plan fftw_2d_plan.cpp fftbenchmarks_utils.h)
fftw_mpi_application(fftw_3d_plan fftw_3d_plan.cpp fftbenchmarks_utils.h)

# AccFFT
accfft_mpi_application(accfft_3d_plan accfft_3d_plan.cpp fftbenchmarks_utils.h)
if (ENABlE_CUDA)
    accfft_mpi_cuda_application(accfft_3d_plan_gpu accfft_3d_plan_gpu.cu fftbenchmarks_utils.h)
endif ()
