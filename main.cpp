#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

class Node {
public:
  std::string name;
};

class Program : public Node {
public:
  std::vector<std::unique_ptr<Node>> program;
};

class ExpressionNode : public Node {};

enum class BinaryOperator : uint32_t { ADD };

class BinaryExpressionNode : ExpressionNode {
public:
  BinaryExpressionNode(uint32_t lhs, BinaryOperator ops, uint32_t rhs)
      : lhs(lhs), op(ops), rhs(rhs) {}

private:
  uint32_t lhs;
  BinaryOperator op;
  uint32_t rhs;
};

class FunctionNode : public Node {
public:
  std::vector<std::unique_ptr<ExpressionNode>> &getExpressions() {
    return expressions;
  }

private:
  std::vector<std::unique_ptr<ExpressionNode>> expressions;
};

uint32_t parsePosition = 0U;

void parseExpression() {}

std::unique_ptr<Node> parseFunction(std::vector<std::string> &tokens) {
  std::unique_ptr<FunctionNode> function = std::make_unique<FunctionNode>();
  parsePosition++;
  std::string_view functionName = tokens.at(parsePosition);
  function->name = functionName;
  parsePosition++; // current is (
  parsePosition++; // current is )
  parsePosition++; // current is {
  parsePosition++;
  std::string_view lhs = tokens.at(parsePosition);
  std::cout << "lhs = " << lhs << std::endl;
  parsePosition++;
  std::string_view operation = tokens.at(parsePosition);
  std::cout << "operator is " << operation << std::endl;
  parsePosition++;
  std::string_view rhs = tokens.at(parsePosition);
  std::cout << "rhs = " << rhs << std::endl;
  std::unique_ptr<BinaryExpressionNode> binary =
      std::make_unique<BinaryExpressionNode>(stoi(lhs), operation, stoi(rhs));
  return function;
}

void parseProgram() {}

void parse(std::vector<std::string> &tokens) {
  std::unique_ptr<Program> program = std::make_unique<Program>();
  parsePosition = 0U;
  std::string_view token = tokens.at(parsePosition);
  if (token == "func") {
    std::cout << "function parse" << std::endl;
    parseFunction(tokens);
  }
}

void tokenizer(std::vector<std::string> &tokens, std::string &program) {
  std::string token;
  for (auto c : program) {
    if (c == ' ' || c == ';' || c == '{' || c == '}' || c == '(' || c == ')' ||
        c == '+') {
      if (!token.empty()) {
        tokens.push_back(token);
        token.clear();
      }
      if (c != ' ') {
        tokens.push_back(std::string(1, c));
      }
    } else {
      token += c;
    }
  }
  if (!token.empty()) {
    tokens.push_back(token);
  }
}

int main(int, char **) {
  std::string input{"func myAdd(){ 3+4 }"};
  std::vector<std::string> tokens;
  tokenizer(tokens, input);
  parse(tokens);
}
