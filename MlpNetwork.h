//MlpNetwork.h

#ifndef MLPNETWORK_H
#define MLPNETWORK_H

#include "Dense.h"
#include "Matrix.h"
#include "Activation.h"
#include "Digit.h"

#define MLP_SIZE 4

//
const matrix_dims img_dims = {28, 28};
const matrix_dims weights_dims[] = {{128, 784},
                                    {64, 128},
                                    {20, 64},
                                    {10, 20}};
const matrix_dims bias_dims[]    = {{128, 1},
                                    {64, 1},
                                    {20, 1},
                                    {10, 1}};

class MlpNetwork
{
 private:
  Matrix* _weights;
  Matrix* _biases;

 public:
  MlpNetwork(Matrix weights[], Matrix biases[]);   // Constructor

  // Operators
  // Parenthesis - Applies the entire network on input returns digit struct
  digit operator()(const Matrix& m) const;
};

#endif // MLPNETWORK_H
