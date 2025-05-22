#include <optional>
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
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/ExecutionEngine/OptUtils.h"
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
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <iostream>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <string>
#include <string_view>
#include <vector>
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
class TestJIT {

private:
    static llvm::Expected<llvm::orc::ThreadSafeModule>
  optimizeModule(llvm::orc::ThreadSafeModule TSM, const llvm::orc::MaterializationResponsibility &R) {
        TSM.withModuleDo([](llvm::Module &M) {
          // Create a function pass manager.
          auto FPM = std::make_unique<llvm::legacy::FunctionPassManager>(&M);

          // Add some optimizations.
          FPM->add(llvm::createInstructionCombiningPass());
          FPM->add(llvm::createReassociatePass());
          FPM->add(llvm::createGVNPass());
          FPM->add(llvm::createCFGSimplificationPass());
          FPM->doInitialization();

          // Run the optimizations over all functions in the module being added to
          // the JIT.
          for (auto &F : M)
              FPM->run(F);
          });

        return std::move(TSM);
    }

private:
    std::unique_ptr<llvm::orc::ExecutionSession> ES;
    llvm::orc::RTDyldObjectLinkingLayer ObjectLayer;
    llvm::orc::IRCompileLayer CompileLayer;

    llvm::orc::IRTransformLayer TransformLayer;

    llvm::DataLayout DL;
    llvm::orc::MangleAndInterner Mangle;

    llvm::orc::JITDylib &Dylib;

public:
    TestJIT() = delete;

    TestJIT(std::unique_ptr<llvm::orc::ExecutionSession> ES,
            llvm::orc::JITTargetMachineBuilder JTMB, llvm::DataLayout DL)
        : ES(std::move(ES)), DL(std::move(DL)), Mangle(*this->ES, this->DL),
          ObjectLayer(*this->ES,
                      []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
          CompileLayer(*this->ES, ObjectLayer,
                       std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(JTMB))),
          TransformLayer(*this->ES, this->CompileLayer, optimizeModule),
          Dylib(this->ES->createBareJITDylib("<main>")) {
        Dylib.addGenerator(
            llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                DL.getGlobalPrefix())));
        if (JTMB.getTargetTriple().isOSBinFormatCOFF()) {
            ObjectLayer.setOverrideObjectFlagsWithResponsibilityFlags(true);
            ObjectLayer.setAutoClaimResponsibilityForObjectSymbols(true);
        }
    }

    ~TestJIT() {
        if (auto err = ES->endSession()) {
            ES->reportError(std::move(err));
        }
    }

    static llvm::Expected<std::unique_ptr<TestJIT> > create() {
        auto EPC = llvm::orc::SelfExecutorProcessControl::Create();
        if (!EPC)
            return EPC.takeError();

        auto ES = std::make_unique<llvm::orc::ExecutionSession>(std::move(*EPC));

        llvm::orc::JITTargetMachineBuilder JTMB(
            ES->getExecutorProcessControl().getTargetTriple());

        auto DL = JTMB.getDefaultDataLayoutForTarget();
        if (!DL)
            return DL.takeError();

        return std::make_unique<TestJIT>(std::move(ES), std::move(JTMB),
                                         std::move(*DL));
    }

    auto &getDataLayout() {
        return this->DL;
    }

    auto& getDylib() const {
        return this->Dylib;
    }

    llvm::Error addModule(llvm::orc::ThreadSafeModule &&TSM, llvm::orc::ResourceTrackerSP RT = nullptr) {
        if (!RT) { RT = Dylib.getDefaultResourceTracker(); }
        auto module = TSM.getModuleUnlocked();
        module->setDataLayout(DL);
        return CompileLayer.add(RT, std::move(TSM));
    }

    llvm::Expected<llvm::orc::ExecutorSymbolDef> lookup(const llvm::StringRef Name) {
        llvm::outs() << "Lookup " << Name << "\n";
        Dylib.dump(llvm::outs());
        return ES->lookup({&Dylib}, Mangle(Name));
    }
};


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

enum class BinaryOperator : uint32_t { ADD, SUB, MUL, DIV };
std::unique_ptr<llvm::Module> dumpLLVMIR(mlir::ModuleOp module, llvm::LLVMContext &llvmContext) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return nullptr;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return nullptr;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return nullptr;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return nullptr;
  }
  llvm::outs() << *llvmModule << "\n";
  return llvmModule;
}


int runJit(mlir::ModuleOp &module) {

}

class BinaryExpressionNode : public ExpressionNode {
public:
  BinaryExpressionNode(uint32_t lhs, BinaryOperator ops, uint32_t rhs)
      : lhs(lhs), op(ops), rhs(rhs) {}
  virtual void codeGen() override {
    // TODO fix this as general purpose generator
    // llvm::DebugFlag = true;
    llvm::ExitOnError exitOnError;
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::MLIRContext context(registry);
    mlir::OpBuilder builder(&context);
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto i32t = mlir::IntegerType::get(&context, 32);
    auto funcType = builder.getFunctionType(std::nullopt, i32t);
    {
      auto function = builder.create<mlir::func::FuncOp>(
          builder.getUnknownLoc(), "testFun", funcType);
      auto entryBlock = function.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      auto intType = mlir::IntegerType::get(&context, 32);
      auto lhsConst = builder.create<mlir::arith::ConstantIntOp>(
          builder.getUnknownLoc(), lhs, intType);
      auto rhsConst = builder.create<mlir::arith::ConstantIntOp>(
          builder.getUnknownLoc(), rhs, intType);
      switch (op) {
        case BinaryOperator::ADD: {
          auto addOp =  builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), lhsConst,
                                          rhsConst);
          builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), addOp.getResult());
          break;
        }
        case BinaryOperator::SUB: {
          auto addOp =  builder.create<mlir::arith::SubIOp>(builder.getUnknownLoc(), lhsConst,
                                          rhsConst);
          builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), addOp.getResult());
          break;
        }
        case BinaryOperator::MUL: {
          auto addOp =  builder.create<mlir::arith::MulIOp>(builder.getUnknownLoc(), lhsConst,
                                          rhsConst);
          builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), addOp.getResult());
          break;
        }
        case BinaryOperator::DIV: {
          auto addOp =  builder.create<mlir::arith::DivSIOp>(builder.getUnknownLoc(), lhsConst,
                                          rhsConst);
          builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), addOp.getResult());
          break;
        }
        default:
        exit(1);
      }

      module.push_back(function);
    }

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
    auto llvm_context = std::make_unique<llvm::LLVMContext>();
    auto llvm_module = dumpLLVMIR(module, *llvm_context);
    auto jit = TestJIT::create();
    if (!jit) {
      llvm::errs() << "Failed to create LLVM jit\n";
    }

    auto err = (*jit)->addModule({std::move(llvm_module), std::move(llvm_context)},
      (*jit)->getDylib().createResourceTracker());
   if (err) {
     llvm::errs() << "Failed to create LLVM jit\n";
   }
    auto symbol = (*jit)->lookup("testFun");
    if (!symbol) {
      llvm::errs() << "Failed to lookup LLVM jit\n";
    }
    auto fun = (int(*)()) symbol->getAddress().getValue();
    llvm::outs() << "jit code address is: 0x" << &fun << "\n";
    std::cout << "jit code call result = " << std::dec << fun() << "\n";
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
  switch (operation[0]) {
    case '+': {
      op = BinaryOperator::ADD;
      break;
    }
    case '-': {
      op = BinaryOperator::SUB;
      break;
    }
    case '*': {
      op = BinaryOperator::MUL;
      break;
    }
    case '/': {
      op = BinaryOperator::DIV;
      break;
    }
    default:
      exit(1);
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
        c == '+' || c == '-' || c == '*' || c == '/') {
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
using namespace llvm;
int main(int, char **) {
  std::string input{"func myAdd(){ 12*21 }"};
  std::vector<std::string> tokens;
  tokenizer(tokens, input);
  auto program = parse(tokens);
  program->codeGen();

  return 0;
}
