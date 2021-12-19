#ifndef C___PROJECT_DENSE_H
#define C___PROJECT_DENSE_H

#include "Activation.h"
#include "Matrix.h"

class Dense
{
 private:
  Matrix _w;
  Matrix _bias;
  ActivationType _act_type;

 public:
  // Constructor
  Dense(const Matrix& w, const Matrix& bias, ActivationType act_type);

  // Methods
  Matrix get_weights() const; // Getter method for weights.
  Matrix get_bias() const; // Getter method for bias.
  ActivationType get_activation() const; // Getter method for activation type.

  // Operators
  // Parenthesis - Applies the layer on input and returns output matrix.
  Matrix operator()(const Matrix& m) const;
};

#endif //C___PROJECT_DENSE_H
