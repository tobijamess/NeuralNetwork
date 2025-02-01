#include <iostream>
#include "neuralnetwork.h"

int main() {
	std::vector<uint32_t> topology = { 2, 3, 1 };
	SimpleNeuralNetwork neuralNetwork(topology, 0.1f);
	std::vector<std::vector<float>> targetInputs {
		{ 0.0f, 0.0f },
		{ 1.0f, 1.0f },
		{ 1.0f, 0.0f },
		{ 0.0f, 1.0f }
	};
	std::vector<std::vector<float>> targetOutputs {
		{ 0.0f },
		{ 0.0f },
		{ 1.0f },
		{ 1.0f }
	};
	uint32_t epoch = 1000000;
	std::cout << "Training started.\n";
	for (uint32_t i = 0; i < epoch; i++) {
		uint32_t index = rand() % 4;
		neuralNetwork.feedForward(targetInputs[index]);
		neuralNetwork.backPropagate(targetOutputs[index]);
	}
	std::cout << "Traning completed.\n";
	for (auto input : targetInputs) {
		neuralNetwork.feedForward(input);
		auto predictions = neuralNetwork.getPrediction();
		std::cout << input[0] << ", " << input[1] << " -> " << predictions[0] << "\n";
	}
	return 0;
}