#include <string>
int main(int argc, char **argv) {
  std::string filepath(argv[1]);
  std::unique_ptr<ProgramIR> programIR =
      protobufToIR(desearializeProtobufFile(filepath));
}
