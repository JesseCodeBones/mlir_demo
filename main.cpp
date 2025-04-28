#define GET_OP_CLASSES
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h.inc"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <iostream>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <string>
#include <string_view>
#include <vector>

class Node {
public:
  std::string name;
  virtual void codeGen() = 0;
  virtual ~Node() = default;
};

class Program : public Node {
public:
  std::vector<std::unique_ptr<Node>> program;
  virtual void codeGen() override {
    for (auto &nodePtr : program) {
      nodePtr->codeGen();
    }
  }
};

class ExpressionNode : public Node {
public:
  virtual void codeGen() override {}
};

enum class BinaryOperator : uint32_t { ADD };

class BinaryExpressionNode : public ExpressionNode {
public:
  BinaryExpressionNode(uint32_t lhs, BinaryOperator ops, uint32_t rhs)
      : lhs(lhs), op(ops), rhs(rhs) {}
  virtual void codeGen() override {
    // TODO fix this as general purpose generator
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::MLIRContext context(registry);
    mlir::OpBuilder builder(&context);
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto funcType = builder.getFunctionType({}, {});
    auto function = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                       "main", funcType);
    auto entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto intType = mlir::IntegerType::get(&context, 32);
    auto lhsConst = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), lhs, intType);
    auto rhsConst = builder.create<mlir::arith::ConstantIntOp>(
        builder.getUnknownLoc(), rhs, intType);
    builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), lhsConst,
                                        rhsConst);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(function);
    // generate IR

    mlir::PassManager pm(module->getName());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(pm.run(module))) {
      std::cerr << "run failed!!\n";
    }
    module.print(llvm::outs());
    auto jitOrError = mlir::ExecutionEngine::create(module);
    std::unique_ptr<mlir::ExecutionEngine> jit = std::move(jitOrError.get());
    llvm::Error error = jit->invoke("main");
  }

private:
  uint32_t lhs;
  BinaryOperator op;
  uint32_t rhs;
};

class FunctionNode : public Node {
public:
  std::vector<std::unique_ptr<ExpressionNode>> &getExpressions() {
    return expressions;
  }
  virtual void codeGen() override {
    for (auto &expression : expressions) {
      expression->codeGen();
    }
  }

private:
  std::vector<std::unique_ptr<ExpressionNode>> expressions;
};

uint32_t parsePosition = 0U;

void parseExpression() {}

std::unique_ptr<Node> parseFunction(std::vector<std::string> &tokens) {
  std::unique_ptr<FunctionNode> function = std::make_unique<FunctionNode>();
  parsePosition++;
  std::string_view functionName = tokens.at(parsePosition);
  function->name = functionName;
  parsePosition++; // current is (
  parsePosition++; // current is )
  parsePosition++; // current is {
  parsePosition++;
  std::string lhs = tokens.at(parsePosition);
  parsePosition++;
  std::string_view operation = tokens.at(parsePosition);
  parsePosition++;
  std::string rhs = tokens.at(parsePosition);
  BinaryOperator op;
  if (operation == "+") {
    op = BinaryOperator::ADD;
  }
  std::unique_ptr<BinaryExpressionNode> binary =
      std::make_unique<BinaryExpressionNode>(
          static_cast<uint32_t>(std::stoul(lhs)), op,
          static_cast<uint32_t>(std::stoul(rhs)));
  function->getExpressions().push_back(std::move(binary));
  return function;
}

void parseProgram() {}

std::unique_ptr<Node> parse(std::vector<std::string> &tokens) {
  std::unique_ptr<Program> program = std::make_unique<Program>();
  parsePosition = 0U;
  std::string_view token = tokens.at(parsePosition);
  if (token == "func") {
    std::cout << "function parse" << std::endl;
    return parseFunction(tokens);
  }
  return nullptr;
}

void tokenizer(std::vector<std::string> &tokens, std::string &program) {
  std::string token;
  for (auto c : program) {
    if (c == ' ' || c == ';' || c == '{' || c == '}' || c == '(' || c == ')' ||
        c == '+') {
      if (!token.empty()) {
        tokens.push_back(token);
        token.clear();
      }
      if (c != ' ') {
        tokens.push_back(std::string(1, c));
      }
    } else {
      token += c;
    }
  }
  if (!token.empty()) {
    tokens.push_back(token);
  }
}

int main(int, char **) {
  std::string input{"func myAdd(){ 3+4 }"};
  std::vector<std::string> tokens;
  tokenizer(tokens, input);
  auto program = parse(tokens);
  program->codeGen();
}
