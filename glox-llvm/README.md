# This is Glox-LLVM Language
This library implements the `glox` language using go for the frontend and LLVM as backend for the IR.

It's a work in progress, extensive implementation is needed.

## TODO
[ ] Tree-walking implementation written in go 
[ ] Dependency diagram and dataflow of the language
[ ] IR lowering & protobuf for serialization of the tree-walking type-cheker
[ ] Protobuf deserialization and LLVM IR codegen

## Inspiration

* The frontend is inspired in [glox](https://github.com/chidiwilliams/glox/tree/main), a go implementation of the tree-walking interpreter from [Crafting Interpreters](https://craftinginterpreters.com/)
* The backend is inspired in the LLVM implementation of (Bolt)[https://github.com/mukul-rathi/bolt/tree/master]

