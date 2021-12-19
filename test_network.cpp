/**
 * Test Mlpnetwork
 * Author: Daniel Winter
 * Date: 04.06.2021
 * Version: 1.0
 */

#include "Matrix.h"
#include "Digit.h"
#include "MlpNetwork.h"
#include <fstream>
#include <iostream>
#include <string>

/* Paths to files */
#define TEST_IMAGES_PATH "dataset/images"
#define TEST_LABELS_PATH "dataset/labels"
#define w1_PATH "parameters/w1"
#define w2_PATH "parameters/w2"
#define w3_PATH "parameters/w3"
#define w4_PATH "parameters/w4"
#define b1_PATH "parameters/b1"
#define b2_PATH "parameters/b2"
#define b3_PATH "parameters/b3"
#define b4_PATH "parameters/b4"

#define INSERT_IMAGE_PATH "Please insert image path:"
#define ERROR_INAVLID_PARAMETER "Error: invalid Parameters file for layer: "
#define ERROR_INVALID_INPUT "Error: Failed to retrieve input. Exiting.."
#define ERROR_INVALID_IMG "Error: invalid image path or size: "

#define ARGS_START_IDX 1
#define ARGS_COUNT 2

using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::ios;

uint32_t swap_endian (uint32_t val)
{
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | (val >> 16);
}

int
run_dataset (const char *image_filename, const char *label_filename, MlpNetwork mlp, int img_num, bool print_fail)
{
  // Open files
  std::ifstream image_file (image_filename, std::ios::in | std::ios::binary);
  if (!image_file.is_open ())
    {
      std::cerr << "Can't open dataset file of images" << endl;
      return EXIT_FAILURE;
    }
  std::ifstream label_file (label_filename, std::ios::in | std::ios::binary);
  if (!image_file.is_open () || !label_file.is_open ())
    {
      std::cerr << "Can't open dataset file of labels" << endl;
      image_file.close ();
      return EXIT_FAILURE;
    }

  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  image_file.read (reinterpret_cast<char *>(&magic), 4);
  magic = swap_endian (magic);
  if (magic != 2051)
    {
      std::cerr << "Invalid file of images in the dataset" << endl;
      return EXIT_FAILURE;
    }

  label_file.read (reinterpret_cast<char *>(&magic), 4);
  magic = swap_endian (magic);
  if (magic != 2049)
    {
      std::cerr << "Invalid file of labels in the dataset" << endl;
      return EXIT_FAILURE;
    }

  image_file.read (reinterpret_cast<char *>(&num_items), 4);
  num_items = swap_endian (num_items);
  label_file.read (reinterpret_cast<char *>(&num_labels), 4);
  num_labels = swap_endian (num_labels);
  if (num_items != num_labels)
    {
      std::cerr << "Invalid dataset: image file nums should equal to label num"
                << endl;
      return EXIT_FAILURE;
    }

  image_file.read (reinterpret_cast<char *>(&rows), 4);
  rows = swap_endian (rows);
  image_file.read (reinterpret_cast<char *>(&cols), 4);
  cols = swap_endian (cols);

  if (img_num > 0)
    {
      num_items = img_num;
    }
  cout << "Testing " << num_items << " images ..." << endl;

  char label;
  char *pixels = new char[rows * cols];

  int error_counter = 0;
  // read each images in the dataset and it's label
  for (int item_id = 0; item_id < num_items; ++item_id)
    {
      Matrix img (rows, cols);

      // read image pixels
      for (int i = 0; i < rows * cols; ++i)
        {
          unsigned char d;
          image_file.read (reinterpret_cast<char *>(&d), 1);
          img[i] = (float) d / 255; // convert  a pixel to [0,1]
        }

      // read label
      label_file.read (&label, 1);
      int correct_value = (int) label;

      Matrix imgVec = img;
      digit output = mlp (imgVec.vectorize ()); // run the network

      if (output.value != correct_value)
        {
          error_counter++;
          if (print_fail)
            {
              std::cout << img;
              cout << "Correct value is: " << correct_value << endl;
              std::cout << "Mlp result: " << output.value <<
                        " at probability: " << output.probability << endl
                        << endl;
            }
        }
    }

  std::cout << "Number of failed classifications: " << error_counter << endl;
  std::cout << "Success rate: : "
            << 100 - 100 * (error_counter / (float) num_items) << "%" << endl;

  delete[] pixels;
  return EXIT_SUCCESS;
}

/**
 * Given a binary file path and a matrix,
 * reads the content of the file into the matrix.
 * file must match matrix in size in order to read successfully.
 * @param filePath - path of the binary file to read
 * @param mat -  matrix to read the file into.
 * @return boolean status
 *          true - success
 *          false - failure
 */
bool readFileToMatrix (const std::string &filePath, Matrix &mat)
{
  std::ifstream is;
  is.open (filePath, std::ios::in | std::ios::binary | std::ios::ate);
  if (!is.is_open ())
    {
      return false;
    }

  long int matByteSize = (long int) mat.get_cols () * mat.get_rows () *
                         sizeof (float);
  if (is.tellg () != matByteSize)
    {
      is.close ();
      return false;
    }

  is.seekg (0, std::ios_base::beg);
  read_binary_file (is, mat);
  is.close ();
  return true;
}

/**
 * Loads MLP parameters from weights & biases paths
 * to Weights[] and Biases[].
 * Exits (code == 1) upon failures.
 * @param paths array of programs arguments, expected to be mlp parameters
 *        path.
 * @param weights array of matrix, weigths[i] is the i'th layer weights matrix
 * @param biases array of matrix, biases[i] is the i'th layer bias matrix
 *          (which is actually a vector)
 */
void loadParameters (string weights_paths[MLP_SIZE],
                     string bias_paths[MLP_SIZE], Matrix weights[MLP_SIZE],
                     Matrix biases[MLP_SIZE])
{
  for (int i = 0; i < MLP_SIZE; i++)
    {
      weights[i] = Matrix (weights_dims[i].rows, weights_dims[i].cols);
      biases[i] = Matrix (bias_dims[i].rows, bias_dims[i].cols);

      std::string weightsPath (weights_paths[i]);
      std::string biasPath (bias_paths[i]);

      if (!(readFileToMatrix (weightsPath, weights[i]) &&
            readFileToMatrix (biasPath, biases[i])))
        {
          std::cerr << ERROR_INAVLID_PARAMETER << (i + 1) << std::endl;
          exit (EXIT_FAILURE);
        }

    }
}

/**
 * Program's main
 * @param argc count of args
 * @param argv args values
 * @return program exit status code
 */
int main (int argc, char **argv)
{

  bool print_fail = false;
  int num_of_images = -1;

  if (argc > 2)
    {
      if (argv[2][0] != '-')
        num_of_images = atoi (argv[2]);
      if (argv[2] == std::string ("-print_fail"))
        print_fail = true;
    }
  if (argc > 1)
    {
      if (argv[1][0] != '-')
        num_of_images = atoi (argv[1]);
      if (argv[1] == std::string ("-print_fail"))
        print_fail = true;
    }

  Matrix weights[MLP_SIZE];
  Matrix biases[MLP_SIZE];
  string weights_paths[] = {w1_PATH, w2_PATH, w3_PATH, w4_PATH};
  string bias_paths[] = {b1_PATH, b2_PATH, b3_PATH, b4_PATH};
  loadParameters (weights_paths, bias_paths, weights, biases);

  MlpNetwork mlp (weights, biases);

  string img_path = TEST_IMAGES_PATH;
  string label_path = TEST_LABELS_PATH;
  return run_dataset (img_path.c_str (), label_path.c_str (), mlp, num_of_images, print_fail);
}
