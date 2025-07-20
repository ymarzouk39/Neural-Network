#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <Eigen/Eigen>

using Eigen::VectorXd, Eigen::seqN, std::function;

VectorXd solve_ode_euler(VectorXd initial_conditions, int input_parameters, int batch_size,
	double end_time, function<VectorXd(double time, VectorXd)>ODE);

VectorXd solve_ode_rk4(VectorXd initial_conditions, int input_parameters, int batch_size,
	double end_time, function<VectorXd(double time, VectorXd)>ODE);

void save_samples(VectorXd samples, std::string file_path);

int main(int argc, char* argv) {
	function<VectorXd(double, VectorXd)> simple_pendulum = [](double time, VectorXd y) {
		VectorXd dy = VectorXd::Zero(y.rows());
		dy(0) = y(1);
		dy(1) = -9.8 * sin(y(0)); // Assuming a simple pendulum equation
		return dy;
		};

	function<VectorXd(double, VectorXd)> func = [](double time, VectorXd y) {
		VectorXd dy = VectorXd::Zero(y.rows());
		dy(0) = y(0);
		return dy;
		};

	VectorXd initial_conditions = VectorXd::Zero(2);
	initial_conditions(0) = 1.0; // Example initial condition
	initial_conditions(1) = 0.0; // Example initial condition

	int input_parameters = static_cast<int>(initial_conditions.rows());
	int batch_size = 1000; // Number of time steps
	double end_time = 10.0; // Total time for the simulation
	VectorXd samples = solve_ode_rk4(initial_conditions, input_parameters, batch_size, end_time, simple_pendulum);
	save_samples(samples, "C:\\Users\\Yousef Marzouk\\Documents\\MATLAB\\ode_samples_pendulum.txt");
}

VectorXd solve_ode_euler(VectorXd initial_conditions, int input_parameters, int batch_size, 
	double end_time, function<VectorXd(double time,VectorXd)>ODE) {
	double time = 0;
	double dt = end_time / batch_size;
	VectorXd samples = VectorXd::Zero(batch_size * input_parameters);
	samples.segment(0, input_parameters) = initial_conditions;
	for (int batch = 1; batch < batch_size; batch++) {
		time += dt;
		int batch_start = batch * input_parameters;
		VectorXd previous_conditions = samples(seqN(batch_start - input_parameters, input_parameters));
		samples(seqN(batch_start, input_parameters)) = 
			previous_conditions + ODE(time, previous_conditions)*dt;
	}
	return samples;
}

VectorXd solve_ode_rk4(VectorXd initial_conditions, int input_parameters, int batch_size,
	double end_time, function<VectorXd(double time, VectorXd)>ODE) {
	double time = 0;
	double dt = end_time / batch_size;
	VectorXd samples = VectorXd::Zero(batch_size * input_parameters);
	samples.segment(0, input_parameters) = initial_conditions;
	for (int batch = 1; batch < batch_size; batch++) {
		time += dt;
		int batch_start = batch * input_parameters;
		double half_step = 0.5 * dt;
		VectorXd previous_conditions = samples(seqN(batch_start - input_parameters, input_parameters));
		VectorXd k1 = ODE(time, previous_conditions);
		VectorXd k2 = ODE(time + half_step, previous_conditions + 0.5 * k1 * dt);
		VectorXd k3 = ODE(time + half_step, previous_conditions + 0.5 * k2 * dt);
		VectorXd k4 = ODE(time + dt, previous_conditions + dt*k3);
		samples(seqN(batch_start, input_parameters)) = 
			previous_conditions + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0);
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