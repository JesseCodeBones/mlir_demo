#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

class Node {
protected:
  std::string name;
};

class Program : public Node {
  std::vector<std::unique_ptr<Node>> program;
};

class ExpressionNode : public Node {};

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

void parseFunction() {}

void parseProgram() {}

void parse(std::vector<std::string> &tokens) {
  parsePosition = 0U;
  std::string_view token = tokens.at(parsePosition);
  if (token == "func") {
    std::cout << "function parse" << std::endl;
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
