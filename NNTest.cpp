#include "NeuralNetwork.cpp"

int main(int argc, char* argv) {
	NeuralNetwork* net = new NeuralNetwork(1, 1, 2, 2, "relu", "relu");

	VectorXd input_values = VectorXd::Constant(1, 1, 1);
	VectorXd expected_values = VectorXd::Constant(1, 1, 7);

	function<double(VectorXd, VectorXd)> least_squares = [](VectorXd input, VectorXd expected) {
		return (input - expected).squaredNorm();
		};

	net->evolution_train(300, 0.01, input_values, expected_values, least_squares);

	std::cout << "Final output: " << net->evaluate(input_values) << std::endl;
	delete net;
}