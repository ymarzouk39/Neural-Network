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
		
		if (layer_count == 1) {
			MatrixXd weight_matrix = MatrixXd::Constant(output_parameters, input_parameters, DEFAULT_WEIGHT);
			weight_matrices.push_back(weight_matrix);
		}
		else {
			MatrixXd input_weights = MatrixXd::Constant(neuron_count, input_parameters, DEFAULT_WEIGHT);
			weight_matrices.push_back(input_weights);

			MatrixXd layer_weights = MatrixXd::Constant(neuron_count, neuron_count, DEFAULT_WEIGHT);
			for (int i = 1; i < layer_count - 1; i++) {
				weight_matrices.push_back(layer_weights);
			}

			MatrixXd output_weights = MatrixXd::Constant(output_parameters, neuron_count, DEFAULT_WEIGHT);
			weight_matrices.push_back(output_weights);
		}
		if (activation_function_in == "relu") {
			activation_function = [this](double input) {
				return this->relu(input);
				};
			derivative_activation_function = [this](double input) {
				return this->relu_derivative(input);
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
				derivative_output_layer_functions.push_back([this](double input) {
					return this->relu_derivative(input);
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
		else if (output_layer_functions_in == "tanh") {
			for (int row = 0; row < output_parameters; row++) {
				output_layer_functions.push_back([this](double input) {
					return this->hypertan(input);
					});
				derivative_output_layer_functions.push_back([this](double input) {
					return this->hypertan_derivative(input);
					});
			}
		}
	}

	void evolution_train(int epoch_count, double diversity_rate, double decay_rate, MatrixXd input_values, 
		MatrixXd expected_values, function<double(VectorXd, VectorXd)> cost_function) {
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

	void grad_descent_train(int epoch_count, int batch_size, double learning_rate, double decay_rate, MatrixXd input_values,
		MatrixXd expected_values, function<double(VectorXd, VectorXd)> cost_function, function<MatrixXd(MatrixXd, MatrixXd)> derivative_cost_function) {
		vector<MatrixXd> activated_values(layer_count + 1, MatrixXd::Zero(neuron_count, batch_size));
		vector<MatrixXd> pre_activated_values(layer_count, MatrixXd::Zero(neuron_count, batch_size));
		activated_values[0] = MatrixXd::Zero(input_parameters, batch_size);
		activated_values[layer_count] = MatrixXd::Zero(output_parameters, batch_size);
		pre_activated_values[layer_count - 1] = MatrixXd::Zero(output_parameters, batch_size);
		vector<MatrixXd> original_weight_matrices = weight_matrices;

		auto follow_gradient = [&](MatrixXd input, MatrixXd expected) {
			original_weight_matrices = weight_matrices;
			activated_values[layer_count] = evaluate_save(input, activated_values, pre_activated_values);
			MatrixXd cost_derivative = derivative_cost_function(activated_values[layer_count], expected);
			MatrixXd layer_values = pre_activated_values[layer_count - 1];
			activate_output_neurons_derivative(layer_values);
			MatrixXd gradient = cost_derivative.array() * layer_values.array();
			MatrixXd transpose = activated_values[layer_count - 1].transpose();
			weight_matrices[layer_count - 1] -= (gradient * transpose) * learning_rate / batch_size;

			for (int layer = layer_count - 1; layer >= 1; layer--) {
				layer_values = pre_activated_values[layer - 1];
				activate_hidden_neurons_derivative(layer_values);
				gradient = (original_weight_matrices[layer].transpose() * gradient).array() * layer_values.array();
				transpose = activated_values[layer - 1].transpose();
				weight_matrices[layer - 1] -= (gradient * transpose) * learning_rate / batch_size;
			}

			return activated_values[layer_count];
		};

		double cost = evaluate_cost(evaluate_many(input_values), expected_values, cost_function);
		std::cout << " Initial cost: " << cost << std::endl;
		for (int epoch = 1; epoch <= epoch_count; epoch++) {
			cost = evaluate_cost(evaluate_many(input_values), expected_values, cost_function);

			learning_rate = learning_rate * exp(-(epoch - 1) * decay_rate);

			for (int batch = 0; batch < input_values.rows()/ batch_size; batch++) {
				MatrixXd input = input_values(seqN(batch * batch_size, batch_size), seqN(0, input_parameters));
				MatrixXd expected = expected_values(seqN(batch * batch_size, batch_size), seqN(0, input_parameters));

				MatrixXd output = follow_gradient(input, expected);

			}
			double new_cost = evaluate_cost(evaluate_many(input_values), expected_values, cost_function);
			if (new_cost < cost) {
				std::cout << "Epoch: " << epoch << ", Cost improved from " << cost << " to " << new_cost << std::endl;
			}
			else {
				std::cout << "Epoch: " << epoch << ", Cost did not improve" << std::endl;
			}
		}
	}

	double evaluate_cost(MatrixXd input_values, MatrixXd expected_values, function<double(VectorXd, VectorXd)> cost_function) {
		double cost = 0;
		VectorXd input_vector = VectorXd::Zero(input_values.cols());
		VectorXd expected_vector = VectorXd::Zero(expected_values.cols());
		for (int batch = 0; batch < input_values.rows(); batch++) {
			input_vector = input_values.row(batch);
			expected_vector = expected_values.row(batch);
			cost += cost_function(input_vector,expected_vector);
		}
		return cost/input_values.rows();
	}

	VectorXd evaluate(VectorXd input) {
		VectorXd layer_values = weight_matrices[0] * input;
		
		for (int layer = 1; layer < layer_count; layer++) {
			activate_hidden_neurons(layer_values);
			layer_values = weight_matrices[layer] * layer_values;
		}
		activate_output_neurons(layer_values);
		return layer_values;
	}
	VectorXd evaluate_save(VectorXd input, vector<VectorXd>&activated_values, vector<VectorXd>& pre_activated_values) {
		VectorXd layer_values = weight_matrices[0] * input;
		activated_values[0] = input;
		for (int layer = 1; layer < layer_count; layer++) {
			pre_activated_values[layer - 1] = layer_values;
			activate_hidden_neurons(layer_values);
			activated_values[layer] = layer_values;
			layer_values = weight_matrices[layer] * layer_values;
		}
		pre_activated_values[layer_count - 1] = layer_values;
		activate_output_neurons(layer_values);
		return layer_values;
	}
	//store values in a vector of matrices where each matrix contains the values of all neurons in a layer for all input samples
	MatrixXd evaluate_save(MatrixXd input, vector<MatrixXd>& activated_values, vector<MatrixXd>& pre_activated_values) {
		MatrixXd layer_values = weight_matrices[0] * input.transpose();
		activated_values[0] = input.transpose();
		for (int layer = 1; layer < layer_count; layer++) {
			pre_activated_values[layer - 1] = layer_values;
			activate_hidden_neurons(layer_values);
			activated_values[layer] = layer_values;
			layer_values = weight_matrices[layer] * layer_values;
		}
		pre_activated_values[layer_count - 1] = layer_values;
		activate_output_neurons(layer_values);
		return layer_values;
	}

	MatrixXd evaluate_many(MatrixXd input) {
		MatrixXd layer_values = weight_matrices[0] * input.transpose();

		for (int layer = 1; layer < layer_count; layer++) {
			activate_hidden_neurons(layer_values);
			layer_values = weight_matrices[layer] * layer_values;
		}
		activate_output_neurons(layer_values);
		return layer_values.transpose();
	}

private:

	void activate_hidden_neurons(VectorXd &input) {
		input.unaryExpr(activation_function);
	}
	void activate_hidden_neurons(MatrixXd& input) {
		input.unaryExpr(activation_function);
	}

	void activate_hidden_neurons_derivative(VectorXd& input) {
		input.unaryExpr(derivative_activation_function);
	}
	void activate_hidden_neurons_derivative(MatrixXd& input) {
		input.unaryExpr(derivative_activation_function);
	}

	void activate_output_neurons(VectorXd &input) {
		for (int row = 0; row < input.rows(); row++) {
			input(row) = output_layer_functions[row](input(row));
		}
	}
	void activate_output_neurons(MatrixXd& input) {
		for (int row = 0; row < input.rows(); row++) {
			input.row(row).unaryExpr(output_layer_functions[row]);
		}
	}

	void activate_output_neurons_derivative(VectorXd& input) {
		for (int row = 0; row < input.rows(); row++) {
			input(row) = derivative_output_layer_functions[row](input(row));
		}
	}
	void activate_output_neurons_derivative(MatrixXd& input) {
		for (int row = 0; row < input.rows(); row++) {
			input.row(row).unaryExpr(derivative_output_layer_functions[row]);
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

	double relu_derivative(double in) {
		if (in <= 0) {
			return 0;
		}
		else {
			return 1;
		}
	}

	double sigmoid(double in) {
		return 1 / (1 + exp(-in));
	}

	double hypertan(double in) {
		return tanh(in);
	}

	double hypertan_derivative(double in) {
		return 1 - pow(tanh(in),2);
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
	vector <function<double(double)>> derivative_output_layer_functions;

};