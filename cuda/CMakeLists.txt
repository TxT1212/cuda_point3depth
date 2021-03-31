project(cuda_boost)

find_package(CUDA QUIET REQUIRED)
set(CMAKE_C_COMPILER /usr/bin/gcc-5)
set(CMAKE_CXX_COMPILER /usr/bin/g++-5) 
set(CUDA_SEPARABLE_COMPILATION ON)
set(
        CUDA_NVCC_FLAGS 
        ${CUDA_NVCC_FLAGS}; 
              -O3 -gencode arch=compute_61,code=sm_61 -rdc=true
              --expt-relaxed-constexpr
)

include_directories(${CUDA_INCLUDE_DIRS})
message(CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS})
# include_directories(/usr/include/crt/)
# include_directories(/usr/local/cuda/include)
# include_directories(/usr/local/cuda-10.2/include/)
# link_directories(/usr/local/cuda-10.2/lib64/)

include_directories("/usr/local/include/eigen3")

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

cuda_add_library(cuda_boost STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

target_link_libraries(
    cuda_boost
    # -lcudadevrt
    # -lcudart
    ${CUDA_LIBRARIES}
   ) 
