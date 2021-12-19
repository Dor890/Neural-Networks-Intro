// Matrix.h
#include <iostream>
#include <fstream>
#include <cmath>

#ifndef MATRIX_H
#define MATRIX_H

using namespace std;

/**
 * @struct matrix_dims
 * @brief Matrix dimensions container. Used in MlpNetwork.h and main.cpp
 */
typedef struct matrix_dims
{
    int rows, cols;
} matrix_dims;

class Matrix
{
 private:
  matrix_dims _dims{};
  float* _matrix;

 public:
  // Constructors and Destructors
  Matrix(int rows, int cols); // Constructor.
  Matrix(); // Default Constructor for 1x1 Matrix.
  Matrix(const Matrix& m); // Copy Constructor.
  ~Matrix(); // Destructor.

  // Class Methods
  int get_rows() const; // Getter method for num of rows in matrix.
  int get_cols() const; // Getter method for num of cols in matrix.
  Matrix& transpose(); // Transposes the Matrix.
  Matrix& vectorize(); // Transforms the matrix into a column vector.
  void plain_print(); // Prints the matrix with existed values.
  Matrix dot(const Matrix& m); // Matrix elements multiplication.
  float norm(); // Returns the Frobenius norm of the given matrix.
  friend Matrix& read_binary_file(istream& is, Matrix& m); // Fills the matrix
  // values from binary file given.

  // Class Operators
  Matrix operator+(const Matrix& m) const; // Matrix Addition.
  Matrix& operator=(const Matrix& m); // Assignment Operator.
  Matrix operator*(const Matrix& m) const; // Matrix multiplication.
  Matrix operator*(float c) const; // Right Scalar multiplication.
  friend Matrix operator*(float c,const  Matrix& m); // Left Scalar mul.
  Matrix& operator+=(const Matrix& m); // Matrix addition accumulation.
  float& operator()(int i, int j); // Parenthesis indexing for changing.
  float operator()(int i, int j) const; // Parenthesis indexing for reading.
  float& operator[](int i); // Brackets indexing for changing.
  float operator[](int i) const; // Brackets indexing for reading.
  friend ostream& operator<<(ostream& os, Matrix& m); // Prints the matrix
  // as asked in the documentation.

};
#endif //MATRIX_H
