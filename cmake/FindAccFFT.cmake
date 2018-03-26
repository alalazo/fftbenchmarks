find_library(ACCFFT_CPU_LIBRARY
    NAMES accfft
    HINTS $ENV{ACCFFT_ROOT}
    PATH_SUFFIXES lib
    DOC "Location of the AccFFT CPU library"
    )

find_library(ACCFFT_CPU_FFTW_LIBRARIES
    NAMES fftw3_omp fftw3
    HINTS $ENV{ACCFFT_FFTW_ROOT}
    PATH_SUFFIXES lib
    DOC "Location of the FFTW library used by AccFFT CPU"
    )

find_path(ACCFFT_INCLUDE_DIRS
    NAMES accfft.h
    HINTS $ENV{ACCFFT_ROOT}
    PATH_SUFFIXES include
    DOC "Location of the AccFFT library header files"
    )