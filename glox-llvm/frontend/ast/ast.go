package ast

import (
	"fmt"
)

type Type interface{}
type ArrayType struct{ Types []Type }
type SingleArray struct {
	Name        string
	GenericArgs []Type
}
type UnionType struct {
	Left  Type
	Right Type
}
type Parameters struct {
	Token Token
	Type  Type
}

type TokenType uint8

const (
	TokenLeftParentesis TokenType = iota
	TokenRightParentesis
	TokenLeftBracelet
	TokenRightBracelet
	TokenLeftBracket
	TokenRightBracket
	TokenComma
	TokenDot
	TokenMinus
	TokenPlus
	TokenSemicolon
	TokenSlash
	TokenStar
	TokenColon
	TokenQuestionMark
	TokenPipe
	TokenEOF
)

type Token struct {
	TokenType TokenType
	Lexer     string
	Literal   interface{}
	Line      int
	Start     int
}

func (t Token) String() string {
	return fmt.Sprintf("%d %s %d", t.TokenType, t.Lexer, t.Literal)
}

type StatementVisitor interface {
	VisitBlockStatement(stmt BlockStatement) interface{}
	VisitClassStatement(stmt ClassStatement) interface{}
	VisitExpressionStatement(stmt ExpressionStatement) interface{}
}

type Statement interface {
	Accept(visitor StatementVisitor) interface{}
	StartLine() int
	EndLine() int
}

type BlockStatement struct {
	statements []Statement
}

type ClassStatement struct {
	Name       Token
	Superclass *VariableExpression
	Init       *FunctionStatement
	Methods    []FunctionStatement
	Fields     []Field
	LineStart  int
	LineEnd    int
}

func (b ClassStatement) StartLine() int { return b.LineStart }
func (b ClassStatement) EndLine() int   { return b.LineEnd }

type Field struct {
	Name  Token
	Value Expression
	Type  Type
}

type FunctionStatement struct {
	Name       Token
	Params     []Parameters
	Body       []Statement
	ReturnType Type
}
