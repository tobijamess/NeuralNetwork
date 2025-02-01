#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <functional>

class Matrix {
public:
	// Members that represent matrix dimensions which store constructor values.
	uint32_t _cols;
	uint32_t _rows;
	std::vector<float> _vals;
public:
	// Default constructor.
	Matrix()
		: _cols(0),
		_rows(0),
		_vals({})
	{
	}
	// Constructor to initialize member variables with parameter values.
	Matrix(uint32_t cols, uint32_t rows) 
		: _cols(cols),
		_rows(rows),
		_vals({})
	{
		// Resize _vals to allocate enough memory to store matrix elements and 
		// initialize with default values.
		_vals.resize(cols * rows, 0.0f);
	}
	
	// Function to apply an activation function to all values of a matrix simultaneously.
	// Used for applying sigmoid and relu to matrices.
	Matrix applyFunction(std::function<float(const float&)> func) {
		Matrix output(_cols, _rows);
		for (uint32_t y = 0; y < output._rows; y++) {
			for (uint32_t x = 0; x < output._cols; x++) {
				output.at(x, y) = func(at(x, y));
			}
		}
		return output;
	}

	// Function to return the value "at" specific row and column indices.
	float &at(uint32_t col, uint32_t row) {
		// Compact the 2D array into a 1D array to find the value "at" an indices.
		return _vals[row * _cols + col];
	}

	// Function to multiply matrix A and B then return matrix C.
	// Check matrix multiplication paint for explanation.
	Matrix multiply(Matrix target) {
		// Use assert to ensure matrix A has the same amount of columns as matrix B
		// has rows to guarantee they can be multiplied.
		assert(_cols == target._rows);
		Matrix output(target._cols, _rows);	
		for (uint32_t y = 0; y < output._rows; y++) {
			for (uint32_t x = 0; x < output._cols; x++) {
				float result = 0.0f;
				for (uint32_t k = 0; k < _cols; k++) {
					result += at(k, y) * target.at(x, k);
				}
				output.at(x, y) = result;
			}
		}
		return output;
	}

	// Function to create a new matrix and multiply each element of the matrix by a scalar value.
	// Used in backpropagation to scale gradients by the learning rate to ensure that adjustments
	// to weights and biases are proportionate to the learning rate.
	Matrix multiplyScalar(float scalar) {
		Matrix output(_cols, _rows);
		for (uint32_t y = 0; y < output._rows; y++) {
			for (uint32_t x = 0; x < output._cols; x++) {
				output.at(x, y) = at(x, y) * scalar;
			}
		}
		return output;
	}

	// Function to perform element-wise multiplication for two matrices of the same dimensions.
	// Used during backpropagation to combine error gradients with the derivative of the activation
	// function. The derivative is calculated element-wise therefore it must be applied to the
	// errors element-wise too.
	Matrix multiplyElements(Matrix target) {
		assert(_rows == target._rows && _cols == target._cols);
		Matrix output(_cols, _rows);
		for (uint32_t y = 0; y < output._rows; y++) {
			for (uint32_t x = 0; x < output._cols; x++) {
				output.at(x, y) = at(x, y) * target.at(x, y);
			}
		}
		return output;
	}

	// Function to add two matrices together.
	// Used to add bias matrices to value matrices.
	Matrix add(Matrix target) {
		// Use assert again to ensure both matrices have the same dimensions.
		assert(_rows == target._rows && _cols == target._cols);
		Matrix output(target._cols, _rows);
		for (uint32_t y = 0; y < output._rows; y++) {
			for (uint32_t x = 0; x < output._cols; x++) {
				output.at(x, y) = at(x, y) + target.at(x, y);
			}
		}
		return output;
	}

	// Function to create a new matrix and add a scalar value to each element of the matrix.
	// Currently unused but could be needed to apply an offset during backpropagation or adjust biases.
	Matrix addScalar(float scalar) {
		Matrix output(_cols, _rows);
		for (uint32_t y = 0; y < output._rows; y++) {
			for (uint32_t x = 0; x < output._cols; x++) {
				output.at(x, y) = at(x, y) + scalar;
			}
		}
		return output;
	}

	// Function to negate each element of a matrix (effectively multiplying each element by -1) to invert them.
	// Used during backpropagation. To calculate the error signal, the network subtracts the predicted
	// values from the target values. It allows the network to compute the direction and magnitude of the 
	// error signal, which is then used to adjust the weights and biases in subsequent steps.
	Matrix negative() {
		Matrix output(_cols, _rows);
		for (uint32_t y = 0; y < output._rows; y++) {
			for (uint32_t x = 0; x < output._cols; x++) {
				output.at(x, y) = -at(x, y);
			}
		}
		return output;
	}

	// Function to return the transpose of a matrix by switching its rows and columns 
	// (_cols become _rows and vice versa). Used during backpropagation to propagate errors to
	// the previous layer and to calculate weight gradients.
	Matrix transpose() {
		Matrix output(_rows, _cols);
		for (uint32_t y = 0; y < _rows; y++) {
			for (uint32_t x = 0; x < _cols; x++) {
				output.at(y, x) = at(x, y);
			}
		}
		return output;
	}
};