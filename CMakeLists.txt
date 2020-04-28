cmake_minimum_required(VERSION 2.8)
project( HumanTracking )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(RPLIDAR_SDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/rplidar_sdk/)
FILE(GLOB RPLIDAR_SDK_SRC
    "${RPLIDAR_SDK_PATH}/src/arch/win32/*.cpp"
    "${RPLIDAR_SDK_PATH}/src/hal/*.cpp"
    "${RPLIDAR_SDK_PATH}/src/*.cpp"
)

add_library(Common Common.h ${RPLIDAR_SDK_SRC})
target_link_libraries(Common ${OpenCV_LIBS})
target_include_directories(Common
    PUBLIC
    ${RPLIDAR_SDK_PATH}/include
    ${RPLIDAR_SDK_PATH}/src
)

add_executable(HumanTracking HumanTracking.cpp HumanTracking.h ${RPLIDAR_SDK_SRC})
target_link_libraries(HumanTracking ${OpenCV_LIBS} Common)
target_include_directories(HumanTracking
    PUBLIC
    ${RPLIDAR_SDK_PATH}/include
    ${RPLIDAR_SDK_PATH}/src
)

add_executable(DataViewer DataViewer.cpp ${RPLIDAR_SDK_SRC})
target_link_libraries(DataViewer ${OpenCV_LIBS} Common)
target_include_directories(DataViewer
    PUBLIC
    ${RPLIDAR_SDK_PATH}/include
    ${RPLIDAR_SDK_PATH}/src
)

add_executable(Test Test.cpp ${RPLIDAR_SDK_SRC})
target_link_libraries(Test ${OpenCV_LIBS} Common)
target_include_directories(Test
    PUBLIC
    ${RPLIDAR_SDK_PATH}/include
    ${RPLIDAR_SDK_PATH}/src
)