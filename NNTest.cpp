#include "NeuralNetwork.cpp"

int main(int argc, char* argv) {
	NeuralNetwork* net = new NeuralNetwork(2, 2, 2, "relu");

	VectorXd m = VectorXd::Constant(2, 1, 1);

	std::cout << net->evaluate(m);

}