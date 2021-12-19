#include "Matrix.h"

#define ERRMATSIZE "Error: matrixes size doesn't fit"
#define ERRREAD "Error: didn't fill whole matrix"
#define ERRCORSIZE "Error: coordinates given doesn't fit"
#define MINVAL 0.1
#define NORMCOLS 1

// Constructor
Matrix::Matrix(int rows, int cols): _dims{rows, cols} {
    _matrix = new float[rows*cols];
    for(int i = 0; i < rows*cols; ++i)
    {
        _matrix[i] = 0;
    }
  }

// // Default Constructor for 1x1 Matrix
Matrix::Matrix(): Matrix(1,1) {}

// Copy Constructor
Matrix::Matrix(const Matrix& m)
{
  _dims.rows = m.get_rows();
  _dims.cols = m.get_cols();
  _matrix = new float[_dims.rows * _dims.cols];
  for(int i = 0; i < _dims.rows * _dims.cols; ++i)
  {
    _matrix[i] = m[i];
  }
}

// Destructor
Matrix::~Matrix ()
{
  delete[] _matrix;
}

// Getter method for num of rows in matrix.
int Matrix::get_rows() const
{
  return _dims.rows;
}

// Getter method for num of cols in matrix.
int Matrix::get_cols() const
{
  return _dims.cols;
}

// Transposes the Matrix.
Matrix& Matrix::transpose()
{
  Matrix trans(_dims.cols, _dims.rows);
  for (int i = 0; i < trans.get_rows(); ++i)
  {
    for (int j = 0; j < trans.get_cols(); ++j)
    {
      trans(i,j) = (*this)(j,i);
    }
  }
  *this = trans;
  return *this;
}

// Transforms the matrix into a column vector.
Matrix& Matrix::vectorize()
{
  _dims.rows = _dims.rows * _dims.cols;
  _dims.cols = NORMCOLS;
  return *this;
}

// Prints the matrix with existed values.
void Matrix::plain_print()
{
  for (int i = 0; i < _dims.rows; ++i)
  {
    for (int j = 0; j < _dims.cols; ++j)
    {
      cout << (*this)(i,j) << " ";
    }
    cout << endl;
  }
}

// Return new Matrix by value of elements multiplication between two
// matrices: given and this.
Matrix Matrix::dot(const Matrix& m)
{
  if (m.get_cols() != _dims.cols || m.get_rows() != _dims.rows)
  {
    cerr << ERRMATSIZE << endl;
    exit(EXIT_FAILURE);
  }
  Matrix new_mat(_dims.rows, _dims.cols);
  for(int i = 0; i < _dims.rows; ++i)
  {
    for (int j = 0; j < _dims.cols; ++j)
    {
      new_mat(i,j) = (*this)(i,j) * m(i,j);
    }
  }
  return new_mat;
}

// Returns the Frobenius norm of the given matrix.
float Matrix::norm()
{
  float sum = 0;
  for(int i = 0; i < _dims.rows; ++i)
  {
    for (int j = 0; j < _dims.cols; ++j)
    {
      sum += pow((*this)(i,j),2);
    }
  }
  return sqrt(sum);
}

// Fills the matrix values from binary file given.
Matrix& read_binary_file(istream& is, Matrix& m)
{
  is.read((char*)m._matrix, m.get_cols() * m.get_rows() * sizeof(float));
  if (is.fail())
  {
    cout << ERRREAD << endl;
    exit(EXIT_FAILURE);
  }
  return m;
}

// Matrix Addition
Matrix Matrix::operator+(const Matrix& m) const
{
  if (m.get_cols() != _dims.cols || m.get_rows() != _dims.rows)
  {
    cerr << ERRMATSIZE << endl;
    exit(EXIT_FAILURE);
  }
  Matrix temp = *this;
  for(int i = 0; i < temp.get_rows(); ++i)
  {
    for (int j = 0; j < temp.get_cols(); ++j)
    {
      temp(i,j) += m(i,j);
    }
  }
  return temp;
}

// Assignment operator
Matrix& Matrix::operator=(const Matrix& m)
{
  if (this == &m) return *this;

  // Free Matrix
  delete[] _matrix;

  // Copy Fields
  _dims.rows = m.get_rows();
  _dims.cols = m.get_cols();
  _matrix = new float[_dims.rows * _dims.cols];
  for(int i = 0; i < _dims.rows * _dims.cols; ++i)
  {
    _matrix[i] = m[i];
  }
  return *this;
}

// Matrix multiplication
Matrix Matrix::operator*(const Matrix& m) const
{
  if (_dims.cols != m.get_rows())
  {
    cerr << ERRMATSIZE << endl;
    exit(EXIT_FAILURE);
  }
  float sum = 0;
  Matrix new_mat(_dims.rows, m.get_cols());
  for(int i = 0; i < new_mat.get_rows(); ++i)
  {
    for (int j = 0; j < new_mat.get_cols(); ++j)
    {
      for (int k = 0; k < _dims.cols; ++k)
      {
        sum += (*this)(i,k) * m(k,j);
      }
      new_mat(i,j) = sum;
      sum = 0;
    }
  }
  return new_mat;
}

// Scalar multiplication on the right
Matrix Matrix::operator*(float c) const
{
  Matrix temp = *this;
  for(int i = 0; i < _dims.cols * _dims.rows; ++i)
  {
    temp[i] *= c;
  }
  return temp;
}

// Scalar multiplication on the left
Matrix operator*(float c, const Matrix& m)
{
  Matrix temp(m);
  for(int i = 0; i < m.get_cols() * m.get_rows(); ++i)
  {
    temp[i] *= c;
  }
  return temp;
}

// Matrix addition accumulation
Matrix& Matrix::operator+=(const Matrix& m)
{
  if (m.get_cols() != _dims.cols || m.get_rows() != _dims.rows)
  {
    cerr << ERRMATSIZE << endl;
    exit(EXIT_FAILURE);
  }
  for(int i = 0; i < _dims.cols * _dims.rows; ++i)
  {
    _matrix[i] += m[i];
  }
  return *this;
}

// Parenthesis indexing for changing (returns by reference)
float& Matrix::operator()(int i, int j)
{
  if (i >= _dims.rows || j >= _dims.cols)
  {
    cerr << ERRCORSIZE << endl;
    exit(EXIT_FAILURE);
  }
  return _matrix[i*_dims.cols + j];
}

// Parenthesis indexing for changing (returns by value)
float Matrix::operator()(int i, int j) const
{
  if (i >= _dims.rows || j >= _dims.cols)
  {
    cerr << ERRCORSIZE << endl;
    exit(EXIT_FAILURE);
  }
  return _matrix[i*_dims.cols + j];
}

// Brackets indexing for changing (returns by reference)
float& Matrix::operator[](int index)
{
  if (index >= _dims.rows * _dims.cols)
  {
    cerr << ERRCORSIZE << endl;
    exit(EXIT_FAILURE);
  }
  return _matrix[index];
}

// Brackets indexing for changing (returns by value)
float Matrix::operator[](int index) const
{
  if (index >= _dims.rows * _dims.cols)
  {
    cerr << ERRCORSIZE << endl;
    exit(EXIT_FAILURE);
  }
  return _matrix[index];
}

// Prints the matrix as asked in the documentation
ostream& operator<<(ostream& os, Matrix& m)
{
  for(int i = 0; i < m.get_rows(); ++i)
  {
    for (int j = 0; j < m.get_cols(); ++j)
    {
      if (m(i,j) >= MINVAL)
      {
        os << "  ";
      }
      else
      {
        os << "**";
      }
    }
    os << endl;
  }
  return os;
}
