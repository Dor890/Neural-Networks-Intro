#include "MlpNetwork.h"

// Constructor
MlpNetwork::MlpNetwork(Matrix weights[], Matrix biases[]):
_weights(weights), _biases(biases) {}

// Parenthesis - Applies the entire network on input returns digit struct
digit MlpNetwork::operator()(const Matrix& m) const
{
  Matrix new_mat(m);
  for (int i = 0; i < MLP_SIZE; ++i)
  {
    if (i < MLP_SIZE-1)
    {
      Dense d(_weights[i], _biases[i], RELU);
      new_mat = d(new_mat);
    }
    else // Last iteration - i = MLP_SIZE-1 = 3
    {
      Dense d(_weights[i], _biases[i], SOFTMAX);
      new_mat = d(new_mat);
    }
  }
  digit ans = {0,new_mat[0]};
  for (int j = 1; j < new_mat.get_rows(); ++j)
  {
    if(new_mat[j] > ans.probability)
    {
      ans.value = j;
      ans.probability = new_mat[j];
    }
  }
  return ans;
}

