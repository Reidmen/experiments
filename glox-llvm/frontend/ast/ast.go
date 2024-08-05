package ast

import "fmt"

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
