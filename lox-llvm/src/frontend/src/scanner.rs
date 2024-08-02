use std::{iter::Peekable, usize};

#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    LeftParentesis,
    RightParentesis,
    LeftBracelet,
    RightBracelet,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Star,
    BanqEqual,
    Equal,
    EqualEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Slash,
    StringLiteral,
    NumberLiteral,
    Identifier,
    And,
    Class,
    Else,
    False,
    For,
    Fun,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Let,
    While,
    EOF,
}

impl std::str::FromStr for TokenKind {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "and" => Ok(TokenKind::And),
            "class" => Ok(TokenKind::Class),
            "else" => Ok(TokenKind::Else),
            "false" => Ok(TokenKind::False),
            "for" => Ok(TokenKind::For),
            "fun" => Ok(TokenKind::Fun),
            "if" => Ok(TokenKind::If),
            "nil" => Ok(TokenKind::Nil),
            "or" => Ok(TokenKind::Or),
            "print" => Ok(TokenKind::Print),
            "super" => Ok(TokenKind::Super),
            "this" => Ok(TokenKind::This),
            "true" => Ok(TokenKind::True),
            "let" => Ok(TokenKind::Let),
            "while" => Ok(TokenKind::While),
            _ => Err(()),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub value: Option<String>,
    pub line: usize,
    pub col: usize,
}

impl Token {
    pub fn kind(&self) -> &TokenKind {
        &self.kind
    }
}

pub struct Tokenizer<I: Iterator<Item = char>> {
    source: Peekable<I>,
    line: usize,
    col: usize,
}
impl<S: Iterator<Item = char>> Tokenizer<S> {
    fn consume(&mut self) -> Option<char> {
        self.col += 1;
        self.source.next()
    }
    fn produce_token(&self, kind: TokenKind) -> Token {
        Token {
            kind,
            value: None,
            line: self.line,
            col: self.col,
        }
    }
    fn produce_token_with_value(&self, kind: TokenKind, value: String) -> Token {
        Token {
            kind,
            value: Some(value),
            line: self.line,
            col: self.col,
        }
    }
}

impl<S: Iterator<Item = char>> Iterator for Tokenizer<S> {
    type Item = Token;
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
