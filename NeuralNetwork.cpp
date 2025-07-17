#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <Eigen/Eigen>

using Eigen::MatrixXd, Eigen::VectorXd;

class NeuralNetwork {
public:

	NeuralNetwork(int input_parameters_in, int layer_count_in, int neuron_count_in, std::string activation_function_in) 
	:input_parameters(input_parameters_in), layer_count(layer_count_in), neuron_count(neuron_count_in) {
		MatrixXd input_weights = MatrixXd::Constant(neuron_count, input_parameters, DEFAULT_WEIGHT);
		weight_matrices.push_back(input_weights);

		MatrixXd layer_weights = MatrixXd::Constant(neuron_count, neuron_count, DEFAULT_WEIGHT);
		for (int i = 1; i < layer_count; i++) {
			weight_matrices.push_back(layer_weights);
		}

		if (activation_function_in == "relu") {
			activation_function = [this](double input) {
				return this->relu(input);
				};
		}
	}

	double evaluate(VectorXd input) {
		VectorXd layer_values = weight_matrices[0] * input;
		activate_neurons(layer_values);
		for (int layer = 1; layer < layer_count; layer++) {
			layer_values = weight_matrices[layer] * layer_values;
			activate_neurons(layer_values);
		}
		return layer_values.sum();
	}






private:

	void activate_neurons(VectorXd &input) {
		for (int row = 0; row < input.rows(); row++) {
			input(row) = activation_function(input(row));
		}
	}

	double relu(double in) {
		if (in <= 0) {
			return 0;
		}
		else {
			return in;
		}
	}

	const double DEFAULT_WEIGHT = 1;
	int input_parameters;
	int layer_count;
	int neuron_count;
	std::function<double(double)> activation_function;
	std::vector<MatrixXd> weight_matrices;

};