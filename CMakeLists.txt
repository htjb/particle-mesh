cmake_minimum_required(VERSION 3.10)
project(particle_mesh)

set(FFTW3_INCLUDE_DIR "/opt/homebrew/opt/fftw/include")
set(FFTW3_LIBRARY "/opt/homebrew/opt/fftw/lib/libfftw3.dylib")  # or .a if static

include_directories(${FFTW3_INCLUDE_DIR})

add_executable(particle_mesh main.cpp)

target_link_libraries(particle_mesh ${FFTW3_LIBRARY})
