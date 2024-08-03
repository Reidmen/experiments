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
