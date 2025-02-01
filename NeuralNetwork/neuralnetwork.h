#pragma once
#include "matrix.h"
#include <cstdlib>

// Sigmoid maps the x input to a number in the range 0 - 1. It's used here to transform the 
// output of each neuron. After computing the weighted sum of the inputs (bias included),
// it ensures the output of the neuron is between 0 and 1.
inline float sigmoid(float x) {
	return 1.0f / (1 + exp(-x));
}
// Derivative of sigmoid, it represents how much the output of the sigmoid function changes,
// with respect to its input. Used during backpropagation to adjust weights.
inline float derivativeSigmoid(float x) {
	return x * (1 - x);
}
// Relu outputs the input directly if its positive, otherwise it outputs 0, this introduces
// non-linearity to help mitigate the vanishing gradient problem.
inline float relu(float x) {
	return x > 0 ? x : 0.0f;
}
// Derivate of relu, it represents how much the output of the relu function changes,
// with respect to its input. Used during backpropagation to adjust weights. The derivative is
// 1 for positive input, and 0 for non-positive input, effectively ignoring negative values.
inline float derivativeRelu(float x) {
	return x > 0 ? 1.0f : 0.0f;
}

// SimpleNeuralNetwork represents a feed-forward neural network, with weights and biases,
// to allow it to perform forward propogation and eventually training through back-propogation. 
// The learning rate controls the magnitude of updates during training.
class SimpleNeuralNetwork {
public:
	// _topology represents the structure of the neural network. Each 
	// element indicates the number of neurons in a corresponding layer 
	// (e.g. [3, 2, 1] defines a network of 3 input, 2 hidden, 1 output layer neurons).
	std::vector<uint32_t> _topology;

	// _weightMatrices stores the weight matrices for connections 
	// between each layer. Each matrix corresponds to weights
	// connecting neurons between two consecutive layers.
	std::vector<Matrix> _weightMatrices;

	// _valueMatrices stores the value matrices for each layer,
	// representing the output values (activations) of neurons.
	// Keeps track of activations (outputs) at each layer during forward propagation.
	std::vector<Matrix> _valueMatrices;

	// _biasMatrices stores the bias matrices for each layer,
	// each matrix contains bias values for neurons in a particular layer.
	// Bias terms are used to adjust the output of neural network layers, alongside weights.
	std::vector<Matrix> _biasMatrices;

	// _learningRate is a scalar value that controls how much the weights are adjusted
	// during training (back-propagation). The learning rate dictates the speed of learning.
	float _learningRate;

public:
	// Constructor to set up the network architecture (define number of neurons in each layer),
	// initialize the weights and biases randomly, prepare storage for neuron activations (outputs),
	// as well as define the rate at which the network learns during training.
	SimpleNeuralNetwork(std::vector<uint32_t> topology, float learningRate = 0.1f) 
		: _topology(topology),
		_weightMatrices({}),
		_valueMatrices({}),
		_biasMatrices({}),
		_learningRate(learningRate)
	{
		// Loop to initialize weights and biases.
		for (uint32_t i = 0; i < topology.size() - 1; i++) {
			// Create matrix object for weights connecting two consecutive layers,
			// with dimensions [neurons in layer i + 1] x [neurons in layer i].
			Matrix weightMatrix(topology[i + 1], topology[i]);
			// Each weight is initialized randomly with rand() / RAND_MAX to get a value between 0 - 1.
			// sqrt(2.0f / topology[i]) is derived from He initialization, which ensures that weights
			// are scaled appropriately based on the number of neurons in the current layer (topology[i]),
			// to mantain stable gradients during training. The lambda initializes each weight as a 
			// product of the normalized random value and the He scaling factor.
			weightMatrix = weightMatrix.applyFunction(
				[i, &topology](const float& f) {
					return ((float)rand() / RAND_MAX) * sqrt(2.0f / topology[i]);
				}
			);
			// The initialized weight matrix is then pushed back into _weightMatrices vector.
			_weightMatrices.push_back(weightMatrix);
			// Create matrix object for biases in each layer (excluding the input layer),
			// with dimensions [neurons in layer i + 1] x 1.
			Matrix biasMatrix(topology[i + 1], 1);
			// Each bias is initialized randomly, the same as the weights.
			biasMatrix = biasMatrix.applyFunction(
				[](const float& f) {
					return (float)rand() / RAND_MAX;
				}
			);
			// The initialized bias matrix is then pushed back into the _biasMatrices vector.
			_biasMatrices.push_back(biasMatrix);
		}
		// _valueMatrices is resized to match the number of layers (topology.size()),
		// allowing it to store values for each layer during forward propogation.
		_valueMatrices.resize(topology.size());
	}

	bool feedForward(std::vector<float> input) {
		if (input.size() != _topology[0]) {
			return false;
		}
		// Convert input (e.g. [1.0, 0.0]) to a value matrix.
		Matrix values(input.size(), 1);
		values._vals = input;

		// For each connection, we assign the values first then we recalculate the value by multiplying by the weight matrices,
		// adding the bias matrices and applying the activation function, which gives us our result.
		for (uint32_t i = 0; i < _weightMatrices.size(); i++) {
			_valueMatrices[i] = values;
			values = values.multiply(_weightMatrices[i]);
			values = values.add(_biasMatrices[i]);
			values = values.applyFunction(sigmoid);
		}
		_valueMatrices[_weightMatrices.size()] = values;
		return true;
	}

	// Function to update weights and biases using backpropagation, which allows the network to learn by adjusting
	// the models parameters based on the error between the predicted and target outputs.
	bool backPropagate(std::vector<float> targetOutput) {
		if (targetOutput.size() != _topology.back()) {
			return false;
		}
		// Create errors matrix with the size and values of the target output.
		Matrix errors(targetOutput.size(), 1);
		errors._vals = targetOutput;
		// Subtract the neural networks output (_valueMatrices.back()) from the target output by
		// adding the negative of the output matrix.
		errors = errors.add(_valueMatrices.back().negative());

		// Loop to iterate backwards through the network from the output layer to the first hidden layer.
		for (int i = _weightMatrices.size() - 1; i >= 0; i--) {
			// Compute the previous layers error by multiplying the current layers error with the
			// transpose of the weight matrix, this propagates the error backwards through the network.
			Matrix prevErrors = errors.multiply(
				_weightMatrices[i].transpose()
			);
			Matrix derivedOutput = _valueMatrices[i + 1].applyFunction(
				derivativeSigmoid
			);
			Matrix gradients = errors.multiplyElements(derivedOutput);
			gradients = gradients.multiplyScalar(_learningRate);
			Matrix weightGradients = _valueMatrices[i].transpose().multiply(gradients);
			_weightMatrices[i] = _weightMatrices[i].add(weightGradients);
			_biasMatrices[i] = _biasMatrices[i].add(gradients);
			errors = prevErrors;
		}
		return true;
	}

	std::vector<float> getPrediction() {
		return _valueMatrices.back()._vals;
	}
};