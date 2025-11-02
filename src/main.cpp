#include <iostream>
#include "neuron.h"
#include "connection.h" 
#include "neuronnet.h" 

void test()
{
	int i = 1;
	float x[]{ 1.4f, 0.3f, 0.1f };
	float y[]{ 0.4f, 0.3f, 0.1f };

	NeuronNet theNN;
	theNN.setModel({ 1, 3, 1 });
	//theNN.init();

	theNN.load("4.txt");

	//theNN.save("4.txt");

	// theNN.predict(x);
/***
	for (i = 1; i < 50; i++)
	{
		x[0] =   0.67; // *5 + 1.0;  // getRandVal(); // 
		y[0] = 4.2 * x[0] * x[0] + 2.2 * x[0] + 2.0;

		std::cout << std::endl;
		std::cout << "    Iteration " << i << std::endl;
		std::cout << "-------------------------" << std::endl;
		std::cout << "  x: " << x[0] << "  y: " << y[0] << std::endl;
		theNN.train(x, y, 0.2f);

		//if (theNN.isFinish())
		//{
		//	theNN.save("5.txt");
		//	break;
		//}

		//theNN.show();
	}
*/
	theNN.save("5.txt");
	theNN.show();
	 

	//theNN.train(x, y, 0.2f);

	//theNN.save("6.txt");

	//theNN.train(x, y, 0.2f);

	//theNN.save("7.txt");

	//theNN.train(x, y, 0.2f);

	//theNN.save("8.txt");


	return;
}

int main(int argc, char* argv[])
{
	std::cout << "Hello World." << std::endl; 

	test();

	return 1;
}