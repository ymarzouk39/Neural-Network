#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <Eigen/Eigen>

using Eigen::VectorXd, Eigen::seqN, std::function;

VectorXd solve_ode_euler(VectorXd initial_conditions, int input_parameters, int batch_size,
	double end_time, function<VectorXd(VectorXd)>ODE);

void save_samples(VectorXd samples, std::string file_path);

int main(int argc, char* argv) {
	function<VectorXd(VectorXd)> ode_function = [](VectorXd x) {
		VectorXd dxdt = -x;
		return dxdt;
		};

	VectorXd initial_conditions = VectorXd::Zero(1);
	initial_conditions(0) = 1.0; // Example initial condition

	int input_parameters = initial_conditions.rows();
	int batch_size = 100; // Number of time steps
	double end_time = 1.0; // Total time for the simulation
	VectorXd samples = solve_ode_euler(initial_conditions, input_parameters, batch_size, end_time, ode_function);
	save_samples(samples, "ode_samples.txt");
}

VectorXd solve_ode_euler(VectorXd initial_conditions, int input_parameters, int batch_size, 
	double end_time, function<VectorXd(VectorXd)>ODE) {
	double dt = end_time / batch_size;
	VectorXd samples = VectorXd::Zero(batch_size * input_parameters);
	samples.segment(0, input_parameters) = initial_conditions;
	for (int batch = 1; batch < batch_size; batch++) {
		int batch_start = batch * input_parameters;
		VectorXd previous_conditions = samples(seqN(batch_start - input_parameters, input_parameters));
		samples(seqN(batch_start, input_parameters)) = 
			previous_conditions + ODE(previous_conditions)*dt;
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