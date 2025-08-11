#include <fstream>
#include "NeuralNetwork.cpp"

MatrixXd read_samples(int sample_size, std::string file_path);
void save_samples(VectorXd samples, std::string file_path);

int main(int argc, char* argv) {

	NeuralNetwork* net = new NeuralNetwork(1, 1, 2, 2, "relu", "relu");

	int batch_size = 100;

	VectorXd time_span = VectorXd::LinSpaced(batch_size, 0, 10);
	MatrixXd input_values = MatrixXd::Zero(batch_size, 1);
	input_values.col(0) = time_span;
	MatrixXd expected_values = read_samples(batch_size, "C:\\Users\\skylo\\OneDrive\\Documents\\MATLAB\\samples.txt");

	function<double(VectorXd, VectorXd)> least_squares = [](VectorXd input, VectorXd expected) {
		double cost = 0;
		for (int row = 0; row < input.rows(); row++) {
			cost += 0.5 * pow(input(row) - expected(row), 2);
		}
		return cost;
	};

	function<MatrixXd(MatrixXd, MatrixXd)> least_squares_derivative = [](MatrixXd input, MatrixXd expected) {
		MatrixXd result = input - expected.transpose();
		return result;
		};

	//net->evolution_train(1000, 0.1, 1e-4, input_values, expected_values, least_squares);
	net->grad_descent_train(10, 10, 0.0001, 1e-4, input_values, expected_values, least_squares,least_squares_derivative);

	MatrixXd output = net->evaluate_many(input_values);
	save_samples(output, "C:\\Users\\skylo\\OneDrive\\Documents\\MATLAB\\nn_samples.txt");
	delete net;
}

MatrixXd read_samples(int sample_size, std::string file_path) {
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