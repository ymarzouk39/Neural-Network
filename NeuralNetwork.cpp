#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <Eigen/Eigen>

using Eigen::MatrixXd, Eigen::VectorXd, std::function, std::vector,
	Eigen::seqN;

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
		else if (activation_function_in == "tanh") {
			activation_function = [this](double input) {
				return this->hypertan(input);
				};
		}
		else if (activation_function_in == "sigmoid") {
			activation_function = [this](double input) {
				return this->sigmoid(input);
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

	void evolution_train(int epoch_count, double diversity_rate, double decay_rate, VectorXd input_values, 
		VectorXd expected_values, function<double(VectorXd, VectorXd)> cost_function) {
		for (int epoch = 1; epoch <= epoch_count; epoch++) {
			double parent_cost = evaluate_cost(evaluate_many(input_values), expected_values, cost_function);
			vector<MatrixXd> parent_weight_matrices = weight_matrices;
			double learning_rate = diversity_rate * exp(-epoch * decay_rate);
			for (int layer = 0; layer < weight_matrices.size(); layer++) {
				weight_matrices[layer] += MatrixXd::Random(weight_matrices[layer].rows(), weight_matrices[layer].cols()) * learning_rate;
			}
			double child_cost = evaluate_cost(evaluate_many(input_values), expected_values, cost_function);
			if (child_cost < parent_cost) {
				std::cout << "Epoch: " << epoch << ", Cost improved from " << parent_cost << " to " << child_cost << std::endl;
			}
			else {
				std::cout << "Epoch: " << epoch << ", Cost did not improve, reverting to parent weights." << std::endl;
				weight_matrices = parent_weight_matrices;
			}
		}
		std::cout << "final model cost: " << evaluate_cost(evaluate_many(input_values), expected_values, cost_function) << std::endl;
	}

	void grad_descent_train(int epoch_count, double diversity_rate, double decay_rate, VectorXd input_values, 
		VectorXd expected_values, function<double(VectorXd, VectorXd)> cost_function, function<VectorXd(VectorXd, VectorXd)> derivative_cost_function) {
		for (int epoch = 1; epoch <= epoch_count; epoch++) {
			MatrixXd activated_values = MatrixXd::Zero(neuron_count, layer_count);
			MatrixXd pre_activated_values = MatrixXd::Zero(neuron_count, layer_count);
			double cost = evaluate_cost(evaluate_save(input_values, activated_values, pre_activated_values), expected_values, cost_function);
			VectorXd cost_derivative = derivative_cost_function(evaluate_many(input_values), expected_values);
			activate_hidden_neurons_derivative(cost_derivative);
			for (int layer = layer_count - 1; layer >= 0; layer--) {
				MatrixXd gradient = cost_derivative * (activated_values.col(layer)).transpose();


			}

		}
	}

	double evaluate_cost(VectorXd input_values, VectorXd expected_values, function<double(VectorXd, VectorXd)> cost_function) {
		double cost = 0;
		for (int batch = 0; batch < input_values.rows() / output_parameters; batch++) {
			int index = batch * output_parameters;
			cost += cost_function(input_values(seqN(index, output_parameters)), expected_values(seqN(index, output_parameters)));
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
	VectorXd evaluate_save(VectorXd input, MatrixXd &activated_values, MatrixXd & pre_activated_values) {
		VectorXd layer_values = weight_matrices[0] * input;

		for (int layer = 1; layer <= layer_count; layer++) {
			pre_activated_values.col(layer - 1) = layer_values;
			activate_hidden_neurons(layer_values);
			activated_values.col(layer - 1) = layer_values;
			layer_values = weight_matrices[layer] * layer_values;
		}
		pre_activated_values.col(layer_count-1) = layer_values;
		activate_output_neurons(layer_values);
		activated_values.col(layer_count - 1) = layer_values;
		return layer_values;
	}

	VectorXd evaluate_many(VectorXd input) {
		VectorXd output = VectorXd::Zero(input.rows());
		for (int batch = 0; batch < output.rows()/output_parameters; batch++) {
			output(seqN(batch * output_parameters, output_parameters)) =
				evaluate(input(seqN(batch * output_parameters, output_parameters)));
		}
		return output;
	}

private:

	void activate_hidden_neurons(VectorXd &input) {
		for (int row = 0; row < input.rows(); row++) {
			input(row) = activation_function(input(row));
		}
	}
	void activate_hidden_neurons_derivative(VectorXd& input) {
		for (int row = 0; row < input.rows(); row++) {
			input(row) = derivative_activation_function(input(row));
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

	double hypertan(double in) {
		return tanh(in);
	}


	const double DEFAULT_WEIGHT = 1;
	int input_parameters;
	int output_parameters;
	int layer_count;
	int neuron_count;
	function<double(double)> activation_function;
	function<double(double)> derivative_activation_function;
	vector<MatrixXd> weight_matrices;
	vector <function<double(double)>> output_layer_functions;

};