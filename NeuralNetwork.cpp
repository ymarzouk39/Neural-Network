#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <functional>
#include <arrayfire.h>


using std::function, std::vector, af::array;
using af::constant, af::randu;
using af::sum, af::pow, af::max, af::sqrt, af::max;
using af::transpose, af::matmul, af:: seq, af::span;
using af::select;

class NeuralNetwork {
public:
	NeuralNetwork(int input_parameters_in, int output_parameters_in, int layer_count_in, int neuron_count_in, 
		std::string activation_function_in, std::string output_layer_functions_in)
	
		:input_parameters(input_parameters_in), output_parameters(output_parameters_in), layer_count(layer_count_in), 
		neuron_count(neuron_count_in) {
		
		vector<vector<array>*> initialize_weights = { &weight_matrices };
		vector<vector<array>*> initialize_biases = { &biases };
		initialize_random_weight_matrices(initialize_weights);
		initialize_random_biases(initialize_biases);

if (activation_function_in == "relu") {
	activation_function = [this](array input) {
		return this->relu(input);
		};
	derivative_activation_function = [this](array input) {
		return this->relu_derivative(input);
		};
}
else if (activation_function_in == "tanh") {
	activation_function = [this](array input) {
		return this->hypertan(input);
		};
	derivative_activation_function = [this](array input) {
		return this->hypertan_derivative(input);
		};
}
else if (activation_function_in == "sigmoid") {
	activation_function = [this](array input) {
		return this->sigmoid(input);
		};
}
if (output_layer_functions_in == "relu") {
	for (int row = 0; row < output_parameters; row++) {
		output_layer_functions.push_back([this](array input) {
			return this->relu(input);
			});
		derivative_output_layer_functions.push_back([this](array input) {
			return this->relu_derivative(input);
			});
	}

}
else if (output_layer_functions_in == "linear") {
	for (int row = 0; row < output_parameters; row++) {
		output_layer_functions.push_back([this](array input) {
			return this->linear(input);
			});
		derivative_output_layer_functions.push_back([this](array input) {
			return this->linear_derivative(input);
			});
	}
}
else if (output_layer_functions_in == "sigmoid") {
	for (int row = 0; row < output_parameters; row++) {
		output_layer_functions.push_back([this](array input) {
			return this->sigmoid(input);
			});
	}
}
else if (output_layer_functions_in == "tanh") {
	for (int row = 0; row < output_parameters; row++) {
		output_layer_functions.push_back([this](array input) {
			return this->hypertan(input);
			});
		derivative_output_layer_functions.push_back([this](array input) {
			return this->hypertan_derivative(input);
			});
	}
}
	}

	void evolution_train(int epoch_count, double diversity_rate, double decay_rate, array input_values,
		array expected_values, function<double(array, array)> cost_function) {
		for (int epoch = 1; epoch <= epoch_count; epoch++) {
			double parent_cost = evaluate_cost(evaluate_many(input_values), expected_values, cost_function);
			vector<array> parent_weight_matrices = weight_matrices;
			vector<array> rand_weights;
			vector<vector<array>*> initialize_weights = { &rand_weights };
			initialize_random_weight_matrices(initialize_weights);
			double learning_rate = diversity_rate * exp(-epoch * decay_rate);
			for (int layer = 0; layer < weight_matrices.size(); layer++) {
				weight_matrices[layer] += rand_weights[layer] * learning_rate;
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

	void grad_descent_train(std::string optimizer, int epoch_count, int batch_size, double learning_rate_in, double decay_rate_1, double decay_rate_2, double weight_decay, array input_values,
		array expected_values, function<double(array, array)> cost_function, function<array(array, array)> derivative_cost_function) {
		//Variable initializations
		vector<array> activated_values(layer_count + 1, constant(0.0, neuron_count, batch_size, f64));
		vector<array> pre_activated_values(layer_count, constant(0.0, neuron_count, batch_size, f64));
		activated_values[0] = constant(0.0, input_parameters, batch_size, f64);
		activated_values[layer_count] = constant(0.0, output_parameters, batch_size, f64);
		pre_activated_values[layer_count - 1] = constant(0.0, output_parameters, batch_size, f64);
		vector<array> original_weight_matrices = weight_matrices;
		array gradient;
		array bias_gradient;
		double learning_rate = learning_rate_in;
		double epsilon = 1e-8;
		int time_step = 1;
		function<void(int)> optimizer_function;
		vector<array> first_moment_weights;
		vector<array> first_moment_biases;
		vector<array> second_moment_weights;
		vector<array> second_moment_biases;

		//lambdas for different optimizers
		auto generic_descent = [&](int layer){
			this->generic_descent(weight_matrices, gradient, layer, learning_rate);
			this->generic_descent(biases, bias_gradient, layer, learning_rate);
			};
		auto RMSProp = [&](int layer) {
			this->RMSProp(weight_matrices, gradient, second_moment_weights, decay_rate_1, epsilon, layer, learning_rate);
			this->RMSProp(biases, bias_gradient, second_moment_biases, decay_rate_1, epsilon, layer, learning_rate);
			};
		auto Adam = [&](int layer) {
			this->Adam(weight_matrices, gradient, first_moment_weights, second_moment_weights, decay_rate_1, decay_rate_2, epsilon, layer, learning_rate, time_step);
			this->Adam(biases, bias_gradient, first_moment_biases, second_moment_biases, decay_rate_1, decay_rate_2, epsilon, layer, learning_rate, time_step);
			};
		auto AdamW = [&](int layer) {
			this->AdamW(weight_matrices, gradient, first_moment_weights, second_moment_weights, decay_rate_1, decay_rate_2, epsilon, layer, learning_rate, time_step, weight_decay);
			this->Adam(biases, bias_gradient, first_moment_biases, second_moment_biases, decay_rate_1, decay_rate_2, epsilon, layer, learning_rate, time_step);
			};

		//Assign optimizer function and Initialize optimizer variables
		if (optimizer == "grad") {
			optimizer_function = generic_descent;
		}
		else if (optimizer == "RMSProp") {
			optimizer_function = RMSProp;
			vector<vector<array>*> initialize_weights = { &second_moment_weights };
			vector<vector<array>*> initialize_bias = { &second_moment_biases };
			initialize_zero_weight_matrices(initialize_weights);
			initialize_zero_biases(initialize_bias);
		}
		else if (optimizer == "Adam") {
			optimizer_function = Adam;
			vector<vector<array>*> initialize_weights = { &first_moment_weights, &second_moment_weights };
			vector<vector<array>*> initialize_bias = { &first_moment_biases, &second_moment_biases };
			initialize_zero_weight_matrices(initialize_weights);
			initialize_zero_biases(initialize_bias);
		}
		else if (optimizer == "AdamW") {
			optimizer_function = AdamW;
			vector<vector<array>*> initialize_weights = { &first_moment_weights, &second_moment_weights };
			vector<vector<array>*> initialize_bias = { &first_moment_biases, &second_moment_biases };
			initialize_zero_weight_matrices(initialize_weights);
			initialize_zero_biases(initialize_bias);
		}

		array cost_derivative;
		array layer_values;
		array loss;
		array transpose;

		

		//gradient descent lambda
		auto follow_gradient = [&](array input, array expected) {
			original_weight_matrices = weight_matrices;
			activated_values[layer_count] = evaluate_save(input, activated_values, pre_activated_values);
			cost_derivative = derivative_cost_function(activated_values[layer_count], expected);
			layer_values = pre_activated_values[layer_count - 1];
			activate_output_neurons_derivative(layer_values);
			loss = cost_derivative * layer_values;
			transpose = af::transpose(activated_values[layer_count - 1]);
			gradient = matmul(loss, transpose) / batch_size;
			bias_gradient = sum(loss, 1) / batch_size;
			//clip_gradient(gradient, 100);
			optimizer_function(layer_count);

			for (int layer = layer_count - 1; layer >= 1; layer--) {
				layer_values = pre_activated_values[layer - 1];
				activate_hidden_neurons_derivative(layer_values);
				loss = matmul(af::transpose(original_weight_matrices[layer]), loss) * layer_values;
				transpose = af::transpose(activated_values[layer - 1]);
				gradient = matmul(loss, transpose) / batch_size;
				bias_gradient = sum(loss, 1) / batch_size;
				//clip_gradient(gradient, 100);
				optimizer_function(layer);

			}
			time_step++;
			return activated_values[layer_count];
		};

		double cost = evaluate_cost(evaluate_many(input_values), expected_values, cost_function);
		std::cout << " Initial cost: " << cost << std::endl;
		seq batch(batch_size);
		
		for (int epoch = 1; epoch <= epoch_count; epoch++) {
			for (int batch = 0; batch < input_values.dims(0)/ batch_size; batch++) {
				array input = input_values(batch, span);
				array expected = expected_values(batch, span);
				batch += batch_size;

				array output = follow_gradient(input, expected);
			}
			if (epoch % 1 == 0) {
				cost = evaluate_cost(evaluate_many(input_values), expected_values, cost_function);
				std::cout << "Epoch: " << epoch << "(" << std::setprecision(3) << static_cast<float>(epoch)/(epoch_count) * 100 << "%), Current cost: " << std::setprecision(4) << cost << std::endl;
			}
		}
	}

	double evaluate_cost(array input_values, array expected_values, function<double(array, array)> cost_function) {
		double cost = 0;
		array input_vector = constant(0.0, input_values.dims(1), f64);
		array expected_vector = constant(0.0, expected_values.dims(1), f64);
		for (int batch = 0; batch < input_values.dims(0); batch++) {
			input_vector = input_values.row(batch);
			expected_vector = expected_values.row(batch);
			cost += cost_function(input_vector,expected_vector);
		}
		return cost/input_values.dims(0);
	}

	array evaluate(array input) {
		array layer_values = matmul(weight_matrices[0], input) + biases[0];
		
		for (int layer = 1; layer < layer_count; layer++) {
			activate_hidden_neurons(layer_values);
			layer_values = matmul(weight_matrices[layer], layer_values) + biases[layer];
		}
		activate_output_neurons(layer_values);
		return layer_values;
	}
	//array evaluate_save(array input, vector<array>&activated_values, vector<array>& pre_activated_values) {
	//	array layer_values = matmul(weight_matrices[0], input) + biases[0];
	//	activated_values[0] = input;
	//	for (int layer = 1; layer < layer_count; layer++) {
	//		pre_activated_values[layer - 1] = layer_values;
	//		activate_hidden_neurons(layer_values);
	//		activated_values[layer] = layer_values;
	//		layer_values = matmul(weight_matrices[layer], layer_values) + biases[layer];
	//	}
	//	pre_activated_values[layer_count - 1] = layer_values;
	//	activate_output_neurons(layer_values);
	//	return layer_values;
	//}
	//store values in a vector of matrices where each matrix contains the values of all neurons in a layer for all input samples
	array evaluate_save(array input, vector<array>& activated_values, vector<array>& pre_activated_values) {
		array layer_values = matmul(weight_matrices[0], transpose(input));
		layer_values += biases[0];
		activated_values[0] = transpose(input);
		for (int layer = 1; layer < layer_count; layer++) {
			pre_activated_values[layer - 1] = layer_values;
			activate_hidden_neurons(layer_values);
			activated_values[layer] = layer_values;
			layer_values = matmul(weight_matrices[layer], layer_values);
			layer_values += biases[layer];
		}
		pre_activated_values[layer_count - 1] = layer_values;
		activate_output_neurons(layer_values);
		return layer_values;
	}

	array evaluate_many(array input) {
		array layer_values = matmul(weight_matrices[0], transpose(input));
		layer_values += biases[0];

		for (int layer = 1; layer < layer_count; layer++) {
			activate_hidden_neurons(layer_values);
			layer_values = matmul(weight_matrices[layer], layer_values);
			layer_values += biases[layer];
		}
		activate_output_neurons(layer_values);
		return transpose(layer_values);
	}

private:

	template<typename MatrixType>
	void initialize_random_weight_matrices(vector<vector<MatrixType>*>& initialize_matrices) {
		if (layer_count == 1) {
			array weight_matrix = randu(output_parameters, input_parameters, f64);
			push_loop(initialize_matrices, weight_matrix);
			return;
		}
		array input_weights = randu(neuron_count, input_parameters, f64);
		push_loop(initialize_matrices, input_weights);


		array layer_weights = randu(neuron_count, neuron_count, f64);
		for (int i = 1; i < layer_count - 1; i++) {
			push_loop(initialize_matrices, layer_weights);
		}

		array output_weights = randu(output_parameters, neuron_count, f64);
		push_loop(initialize_matrices, output_weights);
	}
	template<typename MatrixType>
	void initialize_zero_weight_matrices(vector<vector<MatrixType>*>& initialize_matrices) {
		if (layer_count == 1) {
			array weight_matrix = constant(0.0, output_parameters, input_parameters, f64);
			push_loop(initialize_matrices, weight_matrix);
			return;
		}
		array input_weights = constant(0.0, neuron_count, input_parameters, f64);
		push_loop(initialize_matrices, input_weights);


		array layer_weights = constant(0.0, neuron_count, neuron_count, f64);
		for (int i = 1; i < layer_count - 1; i++) {
			push_loop(initialize_matrices, layer_weights);
		}

		array output_weights = constant(0.0, output_parameters, neuron_count, f64);
		push_loop(initialize_matrices, output_weights);
	}
	template<typename MatrixType>
	void initialize_random_biases(vector<vector<MatrixType>*>& initialize_matrices) {
		for (int layer = 0; layer < layer_count - 1; layer++) {
			array bias = randu(neuron_count, f64);
			push_loop(initialize_matrices, bias);
		}
		push_loop(initialize_matrices, randu(output_parameters, f64));
	}
	template<typename MatrixType>
	void initialize_zero_biases(vector<vector<MatrixType>*>& initialize_matrices) {
		for (int layer = 0; layer < layer_count - 1; layer++) {
			array bias = constant(0.0, neuron_count, f64);
			push_loop(initialize_matrices, bias);
		}
		push_loop(initialize_matrices, constant(0.0, output_parameters, f64));
	}
	template<typename MatrixType>
	void push_loop(vector<vector<MatrixType>*> initialize_matrices, MatrixType reference_matrix) {
		for (int i = 0; i < initialize_matrices.size(); i++) {
			initialize_matrices[i]->push_back(reference_matrix);
		}
	}


	void clip_gradient(array& gradient, double max) { 
		double norm = sum(pow(gradient, 2)).scalar<double>();
		if (norm > max) {
			gradient = gradient * max / norm;
		}
	}

	template<typename MatrixType>
	void generic_descent(vector<MatrixType>& parameter, array& gradient, int layer, double learning_rate) {
		parameter[layer - 1] -= (gradient)*learning_rate;
	}

	template<typename MatrixType>
	void RMSProp(vector<MatrixType>& parameter, array& gradient, vector<array>& running_average, double decay_rate, double epsilon, int layer,
			double learning_rate) {
		/*running_average[layer - 1] = decay_rate * running_average[layer - 1] + (1 - decay_rate) * gradient.array().square().matrix();
		parameter[layer - 1] -= learning_rate * (gradient.array() / (running_average[layer - 1].array().sqrt() + epsilon)).matrix();*/
	}
	template<typename MatrixType>
	void Adam(vector<MatrixType>& parameter, array& gradient, vector<array>& first_moment, vector<array>& second_moment,
		double decay_rate_1, double decay_rate_2, double epsilon, int layer, double learning_rate, int time_step) {
		/*first_moment[layer - 1] = decay_rate_1 * first_moment[layer - 1].array() + (1 - decay_rate_1) * gradient.array();
		second_moment[layer - 1] = decay_rate_2 * second_moment[layer - 1].array() + (1 - decay_rate_2) * gradient.array().square();
		array corrected_first_moment = first_moment[layer - 1] / (1 - pow(decay_rate_1, time_step));
		array corrected_second_moment = second_moment[layer - 1] / (1 - pow(decay_rate_2, time_step));
		parameter[layer - 1] -= learning_rate * (corrected_first_moment.array() / (corrected_second_moment.array().sqrt() + epsilon)).matrix();*/
	}
	template<typename MatrixType>
	void AdamW(vector<MatrixType>& parameter, array& gradient, vector<array>& first_moment, vector<array>& second_moment,
		double decay_rate_1, double decay_rate_2, double epsilon, int layer, double learning_rate, int time_step, double weight_decay) {
		/*first_moment[layer - 1] = decay_rate_1 * first_moment[layer - 1].array() + (1 - decay_rate_1) * gradient.array();
		second_moment[layer - 1] = decay_rate_2 * second_moment[layer - 1].array() + (1 - decay_rate_2) * gradient.array().square();
		array corrected_first_moment = first_moment[layer - 1] / (1 - pow(decay_rate_1, time_step));
		array corrected_second_moment = second_moment[layer - 1] / (1 - pow(decay_rate_2, time_step));
		parameter[layer - 1] = (1 - learning_rate * weight_decay) * parameter[layer - 1] - learning_rate * (corrected_first_moment.array() / (corrected_second_moment.array().sqrt() + epsilon)).matrix();*/
	}

	void activate_hidden_neurons(array &input) {
		input = activation_function(input);
	}


	void activate_hidden_neurons_derivative(array& input) {
		input = derivative_activation_function(input);
	}

	void activate_output_neurons(array &input) {
		for (int row = 0; row < input.dims(0); row++) {
			input(row) = output_layer_functions[row](input(row));
		}
	}

	void activate_output_neurons_derivative(array& input) {
		for (int row = 0; row < input.dims(0); row++) {
			input(row) = derivative_output_layer_functions[row](input(row));
		}
	}

	array relu(const array& in) {
		array mask = in > 0;
		return select(mask, in, 0.0);
	}

	array relu_derivative(const array& in) {
		array mask = in > 0;
		return mask.as(f64);
	}

	array linear(const array& in) {
		return in;
	}

	array linear_derivative(const array& in) {
		return constant(1.0, in.dims(0), in.dims(1), f64);
	}

	array sigmoid(const array in) {
		return 1 / (1 + exp(-in));
	}

	array hypertan(const array& in) {
		return tanh(in);
	}

	array hypertan_derivative(const array& in) {
		return 1 - pow(tanh(in),2);
	}


	const double DEFAULT_WEIGHT = 1;
	const double DEFAULT_BIAS = 0;
	int input_parameters;
	int output_parameters;
	int layer_count;
	int neuron_count;
	function<array(array)> activation_function;
	function<array(array)> derivative_activation_function;
	vector<array> weight_matrices;
	vector<array> biases;
	vector <function<array(array)>> output_layer_functions;
	vector <function<array(array)>> derivative_output_layer_functions;

};