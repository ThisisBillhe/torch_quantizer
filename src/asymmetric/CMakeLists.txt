set(SRCS asymmetric.cpp)

if (CUDA_FOUND)
    list(APPEND SRCS asymmetric.cu)
endif ()


add_library(_C_LIBRARY_ASYMMETRIC STATIC "${SRCS}")
target_link_libraries(_C_LIBRARY_ASYMMETRIC PRIVATE ${TORCH_LIBRARIES})
if (CUDA_FOUND)
    target_link_libraries(_C_LIBRARY_ASYMMETRIC PRIVATE nvidia::cutlass::cutlass nvidia::cutlass::tools::util)
endif ()

list(APPEND _C_LIBRARIES _C_LIBRARY_ASYMMETRIC)
set(_C_LIBRARIES "${_C_LIBRARIES}" PARENT_SCOPE)
