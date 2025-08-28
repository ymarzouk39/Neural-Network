#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <Eigen/Eigen>
#include <cmath>

using Eigen::VectorXd, Eigen::MatrixXd, Eigen::seqN, std::function;

MatrixXd solve_ode_euler(VectorXd initial_conditions, int batch_size,
	double end_time, function<VectorXd(double time, VectorXd)>ODE);

MatrixXd solve_ode_rk4(VectorXd initial_conditions, int batch_size,
	double end_time, function<VectorXd(double time, VectorXd)>ODE);

void save_samples(MatrixXd samples, std::string file_path);

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
	initial_conditions(1) = 2.0; // Example initial condition

	int input_parameters = static_cast<int>(initial_conditions.rows());
	int batch_size = 501; // Number of time steps
	double end_time = 10; // Total time for the simulation
	MatrixXd samples = solve_ode_rk4(initial_conditions, batch_size, end_time, simple_pendulum);
	const double pi = 3.14159265358979323846;
	for (int i = 0; i < samples.rows(); i++) {
		samples(i, 0) = fmod(samples(i, 0) + pi, 2 * pi) - pi;
	}
	save_samples(samples, "C:\\Users\\Yousef Marzouk\\Documents\\MATLAB\\ode_samples.txt");
}

MatrixXd solve_ode_euler(VectorXd initial_conditions, int batch_size,
	double end_time, function<VectorXd(double time,VectorXd)>ODE) {
	double time = 0;
	double dt = end_time / batch_size;
	MatrixXd samples = MatrixXd::Zero(batch_size, initial_conditions.size());
	samples(0,seqN(0,initial_conditions.size())) = initial_conditions;
	for (int batch = 1; batch < batch_size; batch++) {
		time += dt;
		VectorXd previous_conditions = samples(batch - 1, seqN(0, initial_conditions.size()));
		samples(batch, seqN(0, initial_conditions.size())) = previous_conditions + ODE(time, previous_conditions)*dt;
	}
	return samples;
}

MatrixXd solve_ode_rk4(VectorXd initial_conditions, int batch_size,
	double end_time, function<VectorXd(double time, VectorXd)>ODE) {
	double time = 0;
	double dt = end_time / batch_size;
	MatrixXd samples = MatrixXd::Zero(batch_size, initial_conditions.size());
	samples(0, seqN(0, initial_conditions.size())) = initial_conditions;
	for (int batch = 1; batch < batch_size; batch++) {
		time += dt;
		double half_step = 0.5 * dt;
		VectorXd previous_conditions = samples(batch - 1, seqN(0,initial_conditions.size()));
		VectorXd k1 = ODE(time, previous_conditions);
		VectorXd k2 = ODE(time + half_step, previous_conditions + 0.5 * k1 * dt);
		VectorXd k3 = ODE(time + half_step, previous_conditions + 0.5 * k2 * dt);
		VectorXd k4 = ODE(time + dt, previous_conditions + dt*k3);
		samples(batch, seqN(0, initial_conditions.size())) =
			previous_conditions + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0);
	}
	return samples;
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