#include "Activation.h"

// Constructor
Activation::Activation(ActivationType act_type): _act_type(act_type) {}

// Getter method for activation type
ActivationType Activation::get_activation_type ()
{
  return _act_type;
}

// Parenthesis - Applies activation function on input.
Matrix Activation::operator()(const Matrix& m) const
{
  Matrix new_mat(m);
  int length = m.get_cols() * m.get_rows();
  if (_act_type == RELU)
  {
    for (int i = 0; i < length; ++i)
    {
      if (new_mat[i] < 0)
      {
        new_mat[i] = 0;
      }
    }
  }
  else if (_act_type == SOFTMAX)
  {
    float sum = 0;
    float cur_exp;
    for (int i = 0; i < length; ++i)
    {
      cur_exp = exp(new_mat[i]);
      new_mat[i] = cur_exp;
      sum += cur_exp;
    }
    new_mat = (1/sum) * new_mat;
  }
  return new_mat;
}

