CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(PCL_ANNOTATOR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)  
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
find_package(PCL REQUIRED)

IF(NOT WIN32)
    SET(SPECIAL_OS_LIBS "pthread" "X11")
    add_definitions(-Wno-write-strings -fpermissive)
ENDIF()

if(PCL_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(${PCL_INCLUDE_DIRS})
    include_directories(${ZED_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

    link_directories(${ZED_LIBRARY_DIR})
    link_directories(${CUDA_LIBRARY_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})

    add_definitions(-std=c++17 -g -O3)
    list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") ## needed if you install pcl from repo (and libproj-dev)

    if (LINK_SHARED_ZED)
        SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
    else()
        SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
    endif()

	ADD_EXECUTABLE(SAVE_PCD src/save_pcd.cpp)
	TARGET_LINK_LIBRARIES(SAVE_PCD 
                            ${SPECIAL_OS_LIBS} 
                            ${ZED_LIBS} 
                            ${OpenCV_LIBRARIES} 
                            ${PCL_LIBRARIES})                            
	ADD_EXECUTABLE(VIEW_PCD src/annotator.cpp)
	TARGET_LINK_LIBRARIES(VIEW_PCD
                            ${SPECIAL_OS_LIBS} 
                            ${ZED_LIBS} 
                            ${OpenCV_LIBRARIES} 
                            ${PCL_LIBRARIES})                            
endif()
