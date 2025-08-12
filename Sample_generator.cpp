#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Eigen>

using Eigen::VectorXd;

double function(double x);
VectorXd generate_samples(int batch_size, double start_time, double end_time);
void save_samples(VectorXd samples, std::string file_path);

int main(int argc, char* argv) {
	int batch_size = 100; // Number of samples to generate
	VectorXd samples = VectorXd::Zero(batch_size);
	samples = generate_samples(batch_size, 0.0, 10.0); // Generate samples from 0 to 10
	save_samples(samples, "C:\\Users\\skylo\\OneDrive\\Documents\\MATLAB\\samples.txt");
}

double function(double x) {
	return 4*x + 2; // Example function
}

VectorXd generate_samples(int batch_size, double start_time, double end_time) {
	VectorXd samples = VectorXd::Zero(batch_size);
	for (int batch = 0; batch < batch_size; batch++) {
		double time = start_time + batch * (end_time - start_time) / (batch_size - 1);
		samples(batch) = function(time);
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