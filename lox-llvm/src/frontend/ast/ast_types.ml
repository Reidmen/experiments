type loc = Lexing.position

let string_of_loc loc =
  Fmt.str "Line:%d Position:%d" loc.Lexing.pos_lnum
    (loc.Lexing.pos_cnum - loc.Lexing.pos_bol + 1)

module type Id = sig
  type t

  val of_string : string -> t
  val to_string : t -> string
  val ( = ) : t -> t -> bool
end

type mode =
  | Linear
  | ThreadLocal
  | Read
  | Locked
  | ThreadSafe
  | Subordinate
  | Encapsulated

let string_to_mode = function
  | Linear -> "Linear"
  | ThreadLocal -> "ThreadLocal"
  | Read -> "Read"
  | Locked -> "Locked"
  | ThreadSafe -> "ThreadSafe"
  | Subordinate -> "Subordinate"
  | Encapsulated -> "Encapsulated"

type modifier = MutableConst | MutableVar

let string_to_modifier = function
  | MutableConst -> "Const"
  | MutableVar -> "Var"
