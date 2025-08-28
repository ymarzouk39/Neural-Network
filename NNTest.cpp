#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include "NeuralNetwork.cpp"
#include <Eigen/Eigen>

using Eigen::MatrixXd, Eigen::seqN;

MatrixXd read_csv(int sample_size, int input_parameters, std::string file_path);
void save_samples(MatrixXd samples, std::string file_path);
void shuffle_samples(MatrixXd& samples, MatrixXd& input);

int main(int argc, char* argv) {
	
	af::setBackend(AF_BACKEND_OPENCL);
	af::setDevice(1);
	af::info();


	int input_parameters = 1;
	int output_parameters = 1;
	int layer_count = 2;
	int neuron_count = 100;

	int batch_size = 200;
	double start_time = -0.5;
	double end_time = 1.5;

	auto start = std::chrono::high_resolution_clock::now();
	NeuralNetwork* net = new NeuralNetwork(input_parameters, output_parameters, layer_count, neuron_count, "tanh", "linear");

	double dt = (end_time - start_time) / batch_size;
	Eigen::VectorXd time_span = Eigen::VectorXd::LinSpaced(batch_size, start_time, end_time);
	MatrixXd input_data = MatrixXd::Zero(batch_size, 1);
	input_data.col(0) = time_span;
	array eval_samples(input_data.rows(), input_data.cols(), input_data.data());
	MatrixXd expected_data = read_csv(batch_size, input_parameters, "C:\\Users\\Yousef Marzouk\\Documents\\MATLAB\\samples.txt");
	shuffle_samples(expected_data, input_data);
	array expected_values(expected_data.rows(), expected_data.cols(), expected_data.data());
	array input_values(input_data.rows(), input_data.cols(), input_data.data());

	function<double(array, array)> least_squares = [](array input, array expected) {
			
		double cost = 0.5 * sum(pow(input - expected, 2)).scalar<double>();
		return cost;
	};

	function<array(array, array)> least_squares_derivative = [](array input, array expected) {
		array result = input - transpose(expected);
		return result;
		};

	//net->evolution_train(1000, 0.1, 1e-4, input_values, expected_values, least_squares);

	net->grad_descent_train("AdamW", 1000, 2, 0.0001, 0.9, 0.999, 1e-4, input_values, expected_values, least_squares,least_squares_derivative);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> training_time = end - start;
	
	array output = net->evaluate_many(eval_samples);
	MatrixXd output_matrix(output.dims(0), output.dims(1));
	output.host(output_matrix.data());
	save_samples(output_matrix, "C:\\Users\\Yousef Marzouk\\Documents\\MATLAB\\nn_samples.txt");
	std::cout << "Training time: " << training_time.count() << " seconds" << std::endl;
	delete net;
}

MatrixXd read_csv(int sample_size, int input_parameters, std::string file_path) {
	std::ifstream file(file_path);

	if (!file.is_open()) {
		std::cerr << "Error opening file: " << file_path << std::endl;
		return MatrixXd();
	}

	MatrixXd samples = MatrixXd::Zero(sample_size, input_parameters);
	std::string line;
	for (int batch = 0; batch < sample_size; batch++) {
		for (int col = 0; col < input_parameters - 1; col++) {
			std::getline(file, line, ',');
			samples(batch, col) = std::stod(line);
		}
		std::getline(file, line, '\n');
		samples(batch, input_parameters - 1) = std::stod(line);
	}
	return samples;
}

void shuffle_samples(MatrixXd& samples, MatrixXd& input) {
	std::random_device rd; // Provides non-deterministic random numbers
	std::mt19937 gen(rd());

	MatrixXd combined(samples.rows(), samples.cols() + input.cols());
	combined(seqN(0, samples.rows()), seqN(0, samples.cols())) = samples;
	combined(seqN(0, input.rows()), seqN(samples.cols(), input.cols())) = input;
	std::shuffle(combined.rowwise().begin(), combined.rowwise().end(), gen);

	samples = combined(seqN(0, samples.rows()), seqN(0, samples.cols()));
	input = combined(seqN(0, input.rows()), seqN(samples.cols(), input.cols()));
}

void save_samples(MatrixXd samples, std::string file_path) {
	std::ofstream file(file_path);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << file_path << std::endl;
	}
	else {
		for (int line = 0; line < samples.rows() - 1; line++) {
			for (int col = 0; col < samples.cols() - 1; col++) {
				file << samples(line, col) << ",";
			}
			file << samples(line, samples.cols() - 1);
			file << "\n";
		}
		for (int col = 0; col < samples.cols() - 1; col++) {
			file << samples(samples.rows() - 1, col) << ",";
		}
		file << samples(samples.rows() - 1, samples.cols() - 1);
		std::cout << "Samples saved to " << file_path << std::endl;
	}
}
