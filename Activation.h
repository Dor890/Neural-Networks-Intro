//Activation.h
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"

/**
 * @enum ActivationType
 * @brief Indicator of activation function.
 */
enum ActivationType
{
    RELU,
    SOFTMAX
};

class Activation
{
 private:
  ActivationType _act_type;

 public:
  Activation(ActivationType act_type);  // Constructor

  // Class Methods
  ActivationType get_activation_type(); // Getter method for activation type.

  // Operators
  // Parenthesis - Applies activation function on input.
  Matrix operator()(const Matrix& m) const;


};
#endif //ACTIVATION_H
