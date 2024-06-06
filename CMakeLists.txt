cmake_minimum_required(VERSION 3.28)
project(digitor)

set(CMAKE_CXX_STANDARD 23)
include(FetchContent)

FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
)

FetchContent_MakeAvailable(nlohmann_json)

find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)


set(CMAKE_CUDA_ARCHITECTURES native)  # Deprecated

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(digitor src/main.cpp
        src/NeuralNetwork.cpp
        src/NeuralNetwork.h
        src/TrainData.h
        src/matrixor.cu
        src/matrixor.h
)

target_link_libraries(digitor PRIVATE nlohmann_json)
