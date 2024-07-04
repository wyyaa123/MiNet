cmake_minimum_required(VERSION 3.0.0)
project(featurepoint VERSION 0.1.0 LANGUAGES C CXX)

add_executable(main main.cpp)

find_package(OpenCV 3 REQUIRED) 
  
target_link_libraries(main ${OpenCV_LIBS})
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})

