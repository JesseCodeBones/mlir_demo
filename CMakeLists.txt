cmake_minimum_required(VERSION 3.10.0)
set(CMAKE_CXX_STANDARD 17)
project(plus_func VERSION 0.1.0 LANGUAGES C CXX)
# 查找LLVM和MLIR包
find_package(MLIR REQUIRED CONFIG)

# 输出LLVM和MLIR的配置信息
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# 添加LLVM和MLIR的包含目录
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

# 添加LLVM工具
include(HandleLLVMOptions)
include(AddLLVM)

add_executable(plus_func main.cpp)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
# 链接MLIR和LLVM库
target_link_libraries(plus_func PRIVATE
${dialect_libs}
${conversion_libs}
${extension_libs}
MLIRAnalysis
MLIRBuiltinToLLVMIRTranslation
MLIRCallInterfaces
MLIRCastInterfaces
MLIRExecutionEngine
MLIRFunctionInterfaces
MLIRIR
MLIRLLVMCommonConversion
MLIRLLVMToLLVMIRTranslation
MLIRMemRefDialect
MLIRParser
MLIRPass
MLIRSideEffectInterfaces
MLIRTargetLLVMIRExport
MLIRTransforms
)