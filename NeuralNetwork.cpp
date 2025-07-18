#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <Eigen/Eigen>

using Eigen::MatrixXd, Eigen::VectorXd, std::function, std::vector;

class NeuralNetwork {
public:

	NeuralNetwork(int input_parameters_in, int output_parameters_in, int layer_count_in, int neuron_count_in, 
		std::string activation_function_in, std::string output_layer_functions_in)
	
		:input_parameters(input_parameters_in), output_parameters(output_parameters_in), layer_count(layer_count_in), 
		neuron_count(neuron_count_in) {
		
		MatrixXd input_weights = MatrixXd::Constant(neuron_count, input_parameters, DEFAULT_WEIGHT);
		weight_matrices.push_back(input_weights);

		MatrixXd layer_weights = MatrixXd::Constant(neuron_count, neuron_count, DEFAULT_WEIGHT);
		for (int i = 1; i < layer_count; i++) {
			weight_matrices.push_back(layer_weights);
		}

		MatrixXd output_weights = MatrixXd::Constant(output_parameters, neuron_count, DEFAULT_WEIGHT);
		weight_matrices.push_back(output_weights);

		if (activation_function_in == "relu") {
			activation_function = [this](double input) {
				return this->relu(input);
				};
		}
		if (output_layer_functions_in == "relu") {
			for (int row = 0; row < output_parameters; row++) {
				output_layer_functions.push_back([this](double input) {
					return this->relu(input);
					});
			}
		}
		else if (output_layer_functions_in == "sigmoid") {
			for (int row = 0; row < output_parameters; row++) {
				output_layer_functions.push_back([this](double input) {
					return this->sigmoid(input);
					});
			}
		}
	}

	void evolution_train(int epoch_count, double diversity_rate, VectorXd input_values, VectorXd expected_values, function<double(VectorXd, VectorXd)> cost_function) {
		for (int epoch = 1; epoch <= epoch_count; epoch++) {
			double parent_cost = evaluate_cost(evaluate(input_values), expected_values, cost_function);
			vector<MatrixXd> parent_weight_matrices = weight_matrices;
			for (int layer = 0; layer < weight_matrices.size(); layer++) {
				weight_matrices[layer] += MatrixXd::Random(weight_matrices[layer].rows(), weight_matrices[layer].cols()) * diversity_rate;
			}
			double child_cost = evaluate_cost(evaluate(input_values), expected_values, cost_function);
			if (child_cost < parent_cost) {
				std::cout << "Epoch: " << epoch << ", Cost improved from " << parent_cost << " to " << child_cost << std::endl;
			}
			else {
				std::cout << "Epoch: " << epoch << ", Cost did not improve, reverting to parent weights." << std::endl;
				weight_matrices = parent_weight_matrices;
			}
		}
	}

	double evaluate_cost(VectorXd input_values, VectorXd expected_values, function<double(VectorXd, VectorXd)> cost_function) {
		double cost = 0;
		for (int batch = 0; batch < input_values.rows() / output_parameters; batch++) {
			int index = batch * output_parameters;
			cost += cost_function(input_values.segment(index, index + output_parameters), expected_values.segment(index, index + output_parameters));
		}
		return cost;
	}

	VectorXd evaluate(VectorXd input) {
		VectorXd layer_values = weight_matrices[0] * input;
		
		for (int layer = 1; layer <= layer_count; layer++) {
			activate_hidden_neurons(layer_values);
			layer_values = weight_matrices[layer] * layer_values;
		}
		activate_output_neurons(layer_values);
		return layer_values;
	}


private:

	void activate_hidden_neurons(VectorXd &input) {
		for (int row = 0; row < input.rows(); row++) {
			input(row) = activation_function(input(row));
		}
	}

	void activate_output_neurons(VectorXd &input) {
		for (int row = 0; row < input.rows(); row++) {
			input(row) = output_layer_functions[row](input(row));
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

	double sigmoid(double in) {
		return 1 / (1 + exp(-in));
	}


	const double DEFAULT_WEIGHT = 1;
	int input_parameters;
	int output_parameters;
	int layer_count;
	int neuron_count;
	function<double(double)> activation_function;
	vector<MatrixXd> weight_matrices;
	vector <function<double(double)>> output_layer_functions;

};