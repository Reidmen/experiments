use crate::scanner::TokenKind;

/** Abstact Syntax Tree **/
#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    True,
    False,
    Nil,
    String(String),
    Number(f64),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Operator {
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    Plus,
    Minus,
    Multiply,
    Divide,
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Equal => write!(f, "=="),
            Operator::NotEqual => write!(f, "!="),
            Operator::Less => write!(f, "<"),
            Operator::LessOrEqual => write!(f, "<="),
            Operator::Greater => write!(f, ">"),
            Operator::GreaterOrEqual => write!(f, ">="),
            Operator::Plus => write!(f, "+"),
            Operator::Minus => write!(f, "-"),
            Operator::Multiply => write!(f, "*"),
            Operator::Divide => write!(f, "/"),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOperator {
    Not,
    Minus,
}

pub enum Expression {
    Literal(Literal),
    Unary(UnaryOperator, Box<Expression>),
    Binary(Box<Expression>, Operator, Box<Expression>),
    Grouping(Box<Expression>),
    Variable(String),
}

impl From<&TokenKind> for UnaryOperator {
    fn from(token: &TokenKind) -> Self {
        match token {
            TokenKind::Bang => UnaryOperator::Not,
            TokenKind::Minus => UnaryOperator::Minus,
            _ => panic!("can't convert {:?} into UnaryOperator", token),
        }
    }
}
