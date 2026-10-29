# Builds lib/_sim3_solver_cpp*.so (the C++ solver registration.py uses):
#   PY=$(which python)   # the env that will import it, e.g. torch5090
#   cmake -S cpp -B cpp/build -DPython_EXECUTABLE=$PY -Dpybind11_DIR=$($PY -m pybind11 --cmakedir) && cmake --build cpp/build
cmake_minimum_required(VERSION 3.15)
project(sim3_solver_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Eigen3 REQUIRED NO_MODULE)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_sim3_solver_cpp sim3_solver_py.cpp)
target_link_libraries(_sim3_solver_cpp PRIVATE Eigen3::Eigen)
target_compile_options(_sim3_solver_cpp PRIVATE -O3 -march=native -fopenmp-simd)
set_target_properties(_sim3_solver_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib)
