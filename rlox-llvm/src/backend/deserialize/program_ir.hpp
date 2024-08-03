#pragma once

#include "llvm-14/llvm/IR/Type.h"
#include <memory>
#include <string>
#include <vector>

class IRVisitor;

struct TypeIR {
  virtual ~TypeIR() = default;
  virtual llvm::Type *codegen(IRVisitor &visitor) = 0;
}

struct ClassIR {
  std::string className;
  std::vector<std::unique_ptr<TypeIR>> fields;
  std::vector<std::string> vectorTable;
  ClassIR(const frontend::class_defn &classDefinitions);
};

struct ProgramIR {
  std::vector<std::unique_ptr<ClassIR>> classDefinitions;
  std::vector<std::unique_ptr<FunctionIR>> functionDefinitions;
  std::vector<std::unique_ptr<ExpressionIR>> mainExpression;

  ProgramIR(const frontend::program &program);
  ProgramIR(){};
};
