package parser

import (
	"frontend/ast"
	"io"
)

// Parser parses a flat list of tokens
// into an AST representation of the source program
type Parser struct {
	tokens   []ast.Token
	current  int
	loop     int
	hadError bool
	stdError io.Writer
}

/*
*
Parser grammar:

	program => declaration* EOF
	declaration => classDeclaration, funcDeclaration, varDeclaration, TypeDeclaration, statement
	classDeclaration => "class" IDENTIFIER ( "<" IDENTIFIER ) ? "{" (method | field)* "}"
	TODO!

*
*/
func (p *Parser) Parse() (ast.Statement, bool) {
	var statements []ast.Statement
	for !p.isAtEnd() {
		statement := p.declaration()
		statements = append(statements, statement)
	}
	return statements, p.hadError
}
