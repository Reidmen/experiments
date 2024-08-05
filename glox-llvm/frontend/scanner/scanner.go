package scanner

import (
	"fmt"
	"frontend/ast"
	"io"
)

type Scanner struct {
	start    int
	current  int
	line     int
	source   string
	tokens   []ast.Token
	stdError io.Writer
}

func newScanner(source string, stdError io.Writer) *Scanner {
	return &Scanner{source: source, stdError: stdError}
}

func (s *Scanner) scanTokens() []ast.Token {
	for !s.isAtEnd() {
		s.start = s.current
		s.scanTokens()
	}

	s.tokens = append(s.tokens, ast.Token{TokenType: ast.TokenEOF, Line: s.line})
	return s.tokens
}

func (s *Scanner) isAtEnd() bool       { return s.current >= len(s.source) }
func (s *Scanner) isDigit(r rune) bool { return r >= '0' && r <= '9' }

func (s *Scanner) advance() rune {
	curr := rune(s.source[s.current])
	s.current++
	return curr
}

func (s *Scanner) addToken(tokenType ast.TokenType) {
	s.addTokenWithLiteral(tokenType, nil)
}

func (s *Scanner) addTokenWithLiteral(tokenType ast.TokenType, literal interface{}) {
	text := s.source[s.start:s.current]
	token := ast.Token{
		TokenType: tokenType,
		Lexer:     text,
		Literal:   literal,
		Line:      s.line,
		Start:     s.start,
	}
	s.tokens = append(s.tokens, token)
}

func (s *Scanner) scanToken() {
	char := s.advance()
	switch char {
	case '(':
		s.addToken(ast.TokenLeftParentesis)
	case ')':
		s.addToken(ast.TokenRightParentesis)
	case '{':
		s.addToken(ast.TokenLeftBracelet)
	case '}':
		s.addToken(ast.TokenRightBracelet)
	case '[':
		s.addToken(ast.TokenLeftBracket)
	case ']':
		s.addToken(ast.TokenRightBracket)
	case ',':
		s.addToken(ast.TokenComma)
	case '.':
		s.addToken(ast.TokenDot)
	case '-':
		s.addToken(ast.TokenMinus)
	case '+':
		s.addToken(ast.TokenPlus)
	case ';':
		s.addToken(ast.TokenSemicolon)
	case ':':
		s.addToken(ast.TokenColon)
	}
}

func (s *Scanner) error(message string) {
	_, _ = s.stdError.Write([]byte(fmt.Sprintf("[line %d] Error: %s]n", s.line, message)))
}
