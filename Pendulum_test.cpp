#include <fstream>
#include <algorithm>
#include <random>
#include "NeuralNetwork.cpp"

MatrixXd read_csv(int sample_size, std::string file_path);
void save_samples(VectorXd samples, std::string file_path);
void shuffle_samples(MatrixXd& samples, MatrixXd& input);

int main(int argc, char* argv) {

	NeuralNetwork* net = new NeuralNetwork(1, 1, 3, 20, "tanh", "linear");

	int batch_size = 1000;
	double start_time = 0;
	double end_time = 10;

	VectorXd time_span = VectorXd::LinSpaced(batch_size, start_time, end_time);
	MatrixXd input_values = MatrixXd::Zero(batch_size, 1);
	input_values.col(0) = time_span;
	MatrixXd eval_samples = input_values;
	MatrixXd expected_values = read_csv(batch_size, "C:\\Users\\skylo\\OneDrive\\Documents\\MATLAB\\ode_samples.txt");
	//shuffle_samples(expected_values, input_values);

	function<double(VectorXd, VectorXd)> least_squares = [](VectorXd input, VectorXd expected) {

		double cost = 0.5 * (input - expected).squaredNorm();
		return cost;
		};

	function<MatrixXd(MatrixXd, MatrixXd)> least_squares_derivative = [](MatrixXd input, MatrixXd expected) {
		MatrixXd result = input - expected.transpose();
		return result;
		};

	//net->evolution_train(1000, 0.1, 1e-4, input_values, expected_values, least_squares);

	net->grad_descent_train("AdamW", 500, 2, 0.0002, 0.9, 0.999, 1e-4, input_values, expected_values, least_squares, least_squares_derivative);

	MatrixXd output = net->evaluate_many(eval_samples);
	save_samples(output, "C:\\Users\\skylo\\OneDrive\\Documents\\MATLAB\\nn_samples.txt");
	delete net;
}

MatrixXd read_csv(int sample_size, std::string file_path) {
	std::ifstream file(file_path);

	if (!file.is_open()) {
		std::cerr << "Error opening file: " << file_path << std::endl;
		return MatrixXd();
	}

	MatrixXd samples = MatrixXd::Zero(sample_size, 1);
	for (int batch = 0; batch < sample_size; batch++) {
		file >> samples(batch, 0);
	}
	return samples;
}

void shuffle_samples(MatrixXd& samples, MatrixXd& input) {
	std::random_device rd;
	std::mt19937 gen(rd());

	MatrixXd combined(samples.rows(), samples.cols() + input.cols());
	combined(seqN(0, samples.rows()), seqN(0, samples.cols())) = samples;
	combined(seqN(0, input.rows()), seqN(samples.cols(), input.cols())) = input;
	std::shuffle(combined.rowwise().begin(), combined.rowwise().end(), gen);

	samples = combined(seqN(0, samples.rows()), seqN(0, samples.cols()));
	input = combined(seqN(0, input.rows()), seqN(samples.cols(), input.cols()));
}

void save_samples(VectorXd samples, std::string file_path) {
	std::ofstream file(file_path);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << file_path << std::endl;
	}
	else {
		for (int line = 0; line < samples.rows() - 1; line++) {
			file << samples(line);
			file << "\n";
		}
		file << samples(samples.rows() - 1);
		std::cout << "Samples saved to " << file_path << std::endl;
	}
}
