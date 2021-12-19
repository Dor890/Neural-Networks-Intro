#include "Dense.h"

// Constructor
Dense::Dense(const Matrix& w, const Matrix& bias, ActivationType act_type):
  _w(w), _bias(bias), _act_type(act_type) {}

// Getter method for weights
Matrix Dense::get_weights () const
{
  return _w;
}

// // Getter method for bias
Matrix Dense::get_bias () const
{
  return _bias;
}

// Getter method for activation type
ActivationType Dense::get_activation () const
{
  return _act_type;
}

// Parenthesis - Applies the layer on input and returns output matrix
Matrix Dense::operator()(const Matrix& m) const
{
  Matrix new_mat = _w * m;
  new_mat += _bias;
  Activation at(_act_type);
  return at(new_mat);
}