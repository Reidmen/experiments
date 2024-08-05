#pragma once

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

enum BinaryOperation {
  BinaryOperationPlus,
  BinaryOperationMinus,
  BinaryOperationMult,
  BinaryOperationIntDiv,
  BinaryOperationRem,
  BinaryOperationLessThan,
  BinaryOperationLessThanEq,
  BinaryOperationGreaterThan,
  BinaryOperationGreaterThanEq,
  BinaryOperationAnd,
  BinaryOperationOr,
  BinaryOperationEq,
  BinaryOperationNotEq,
};

enum UnaryOperation {
  UnaryOperationNot,
  UnaryOperationNeg,
};

struct TypeIntIR;
struct TypeClassIR;
struct TypeVoidIR;
struct TypeBoolIR;

class IRVisitor {
public:
  virtual llvm::Type *codegen(const TypeIntIR &typeIR) = 0;
  virtual llvm::Type *codegen(const TypeClassIR &typeIR) = 0;
  virtual llvm::Type *codegen(const TypeVoidIR &typeIR) = 0;
  virtual llvm::Type *codegen(const TypeBoolIR &typeIR) = 0;
};

struct TypeIR {
  virtual ~TypeIR() = default;
  virtual llvm::Type *codegen(IRVisitor &visitor) = 0;
};

struct TypeIntIR : public TypeIR {
  virtual llvm::Type *codegen(IRVisitor &visitor) override;
};

struct TypeClassIR : public TypeIR {
  std::string className;
  TypeClassIR(const std::string &name) : className(name) {}
  virtual llvm::Type *codegen(IRVisitor &visitor) override;
};

// TODO TypeClass TypeVoidIR, TypeBoolIR
//

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

/* Identifier IR */
struct IdentifierIR {
  std::string varName;
  virtual ~IdentifierIR() = default;
  virtual llvm::Value *codegen(IRVisitor &visitor) = 0;
};

std::unique_ptr<IdentifierIR>
deserializeIdentifier(const frontend::identifier &identifier) {};

struct IdentifierVarIR : public IdentifierIR {
  IdentifierVarIR(const std::string &name);
  virtual llvm::Value *codegen(IRVisitor &visitor) override;
};

class IRCodegenVisitor : public IRVisitor {
  const int NUM_RESERVED_FIELDS = 4;

protected: /* Used by tester */
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::IRBuilder<>> builder;
  std::unique_ptr<llvm::Module> module;
  std::map<std::string, llvm::AllocaInst *> varEnv;

public:
  IRCodegenVisitor();

  void configureTarget();
  void dumpLLVMIR();
  std::string dumpLLVMIRToString();
  void codegenProgram(const ProgramIR &program);
  void codegenMainExpression(
      const std::vector<std::unique_ptr<ExpressionIR>> &mainExpression);
  llvm::Type *codegenPthreadTy();
};
