type loc = Lexing.position

module type Id = sig
  type t

  val of_string : string -> t
  val to_string : t -> string
  val ( = ) : t -> t -> bool
end

module Var_name : Id
module Class_name : Id
module Capacity_name : Id
module Field_name : Id
module Method_name : Id
module Function_name : Id

type mode =
  | Linear
  | ThreadLocal
  | Read
  | Locked
  | ThreadSafe
  | Subordinate
  | Encapsulated

type modifier = MutableConst | MutableVar
type borrowed_reference = Borrowed
type generic = Generic

type type_expr =
  | TypeExprInt
  | TypeExprClass of Class_name.t * type_expr option
  | TypeExprVoid
  | TypeExprBool
  | TypeExprGeneric

type field_definition =
  | TypeField of modifier * type_expr * Field_name.t * Capacity_name.t list

type capability = TypeCapability of mode * Capacity_name.t

type parameter =
  | TypeParameter of
      type_expr
      * Var_name.t
      * Capacity_name.t list option
      * borrowed_reference option

val get_parameters_types : parameter list -> type_expr list

(* Binary operators for expression *)
type binary_operator =
  | BinaryOpPlus
  | BinaryOpMinus
  | BinaryOpMult
  | BinaryOpIntDiv
  | BinaryOpRem
  | BinaryOpLessThan
  | BinaryOpLessThanEq
  | BinaryOpGreaterThan
  | BinaryOpGreaterThanEq
  | BinaryOpAnd
  | BinaryOpOr
  | BinaryOpEq
  | BinaryOpNotEq

val string_to_loc : loc -> string
val string_to_mode : mode -> string
val string_of_capability : capability -> string
val string_of_modifier : modifier -> string
val string_of_type : type_expr -> string
val string_of_binary_operator : binary_operator -> string
