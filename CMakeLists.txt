CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(looping)

FIND_PACKAGE(OpenCV 4.2.0 REQUIRED PATHS "/shared/foss/opencv-4.2.0_install/lib/cmake/opencv4")

ADD_EXECUTABLE(looping
"${CMAKE_SOURCE_DIR}/looping.cpp" 
)
TARGET_LINK_LIBRARIES(looping ${OpenCV_LIBS})
