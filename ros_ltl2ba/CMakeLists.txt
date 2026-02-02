cmake_minimum_required(VERSION 3.5)
project(ltl2ba)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Enable C language
enable_language(C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNXT")


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Add the library
add_library(ltl2ba STATIC
    src/parse.c
    src/lex.c
    src/trans.c
    src/buchi.c
    src/set.c
    src/mem.c
    src/rewrt.c
    src/cache.c
    src/alternating.c
    src/generalized.c
    src/ltl2ba.h
)

# Add the executable
add_executable(main src/main.c src/ltl2ba.h)
set_target_properties(main PROPERTIES OUTPUT_NAME ltl2ba)
ament_target_dependencies(main rclcpp)
target_link_libraries(main ltl2ba)

# Install the library and executable
install(TARGETS
    ltl2ba
    main
    DESTINATION lib/${PROJECT_NAME}/
  )

ament_package()
