#include <fstream>
#include "NeuralNetwork.cpp"

VectorXd read_samples(int sample_size, std::string file_path);
void save_samples(VectorXd samples, std::string file_path);

int main(int argc, char* argv) {
	NeuralNetwork* net = new NeuralNetwork(1, 1, 1, 5, "tanh", "relu");

	VectorXd input_values = VectorXd::LinSpaced(100, 0, 10);
	VectorXd expected_values = read_samples(100, "C:\\Users\\Yousef Marzouk\\Documents\\MATLAB\\samples.txt");

	function<double(VectorXd, VectorXd)> least_squares = [](VectorXd input, VectorXd expected) {
		double cost = 0;
		for (int row = 0; row < input.rows(); row++) {
			cost += pow(input(row) - expected(row), 2);
		}
		return cost;
	};

	net->evolution_train(10000, 1, 0.0001, input_values, expected_values, least_squares);

	VectorXd output = net->evaluate_many(input_values);
	save_samples(output, "C:\\Users\\Yousef Marzouk\\Documents\\MATLAB\\nn_samples.txt");
	delete net;
}

VectorXd read_samples(int sample_size, std::string file_path) {
	std::ifstream file(file_path);

	if (!file.is_open()) {
		std::cerr << "Error opening file: " << file_path << std::endl;
		return VectorXd();
	}
	
	VectorXd samples = VectorXd::Zero(sample_size);
	for (int batch = 0; batch < sample_size; batch++) {
		file >> samples(batch);
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