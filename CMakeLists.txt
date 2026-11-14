cmake_minimum_required(VERSION 3.22.1)
project("llama_edusoho" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimasi kompilasi ARM64 NEON / DotProd untuk matriks Q4_K_M pada CPU Android
if(ANDROID_ABI STREQUAL "arm64-v8a")
    add_compile_options(-O3 -ffast-math -funroll-loops)
else()
    add_compile_options(-O3 -ffast-math)
endif()

# Deteksi otomatis apabila submodule llama.cpp sudah ditempatkan di app/src/main/cpp/llama.cpp
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    message(STATUS "EduSoho GGUF: Submodule llama.cpp ditemukan, mengompilasi penuh ggml + llama")
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    add_subdirectory(llama.cpp)

    add_library(
        llama_edusoho
        SHARED
        llama_edusoho_jni.cpp
    )

    target_compile_definitions(llama_edusoho PRIVATE EDUSOHO_HAS_LLAMA_CPP=1)
    target_include_directories(
        llama_edusoho
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
    )

    find_library(log-lib log)
    find_library(android-lib android)

    target_link_libraries(
        llama_edusoho
        llama
        ggml
        ${log-lib}
        ${android-lib}
    )
else()
    message(STATUS "EduSoho GGUF: Mengompilasi libllama_edusoho.so dalam mode Standalone Native GGUF mmap + Contextual Engine")
    add_library(
        llama_edusoho
        SHARED
        llama_edusoho_jni.cpp
    )

    find_library(log-lib log)
    find_library(android-lib android)

    target_link_libraries(
        llama_edusoho
        ${log-lib}
        ${android-lib}
    )
endif()
