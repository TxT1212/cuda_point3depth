CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
#// 项目名称
PROJECT(Demo)

add_subdirectory(cuda)
set (EXTRA_LIBS ${EXTRA_LIBS} cuda_boost)
find_package(CUDA QUIET REQUIRED)
message(CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS})
add_definitions(-DUNIX)
add_compile_options(-Wno-deprecated-declarations)
add_compile_options(-Wno-unused-result)
add_definitions(-O3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
link_libraries(${PCL_LIBRARIES_DIRS})
add_definitions(${PCL_DEFINITIONS})
message(${PCL_INCLUDE_DIRS})
include_directories("/usr/local/include/eigen3")
# include_directories("/usr/include/crt/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE( ${PROJECT_NAME}
    # main_ts.cc
    main.cpp
)
# CUDA_ADD_EXECUTABLE( project_ts
#     main_ts.cc
# )
add_executable(reprojectError reprojectError.cc)

target_link_libraries(reprojectError
    ${OpenCV_LIBRARIES}
)
#// 链接外部库，需要包含cudalib
# TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
#     ${CUDA_LIBRARIES}
#     -lcudadevrt
#     -lcudart
# )
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${EXTRA_LIBS})
# target_link_libraries(project_ts ${PCL_LIBRARIES})
# target_link_libraries(project_ts ${OpenCV_LIBRARIES})
# target_link_libraries(project_ts ${EXTRA_LIBS})