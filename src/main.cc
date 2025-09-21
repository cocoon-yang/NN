#include <iostream>
#include "nn.h"

void testNN()
{   
	float x[]{ 1.4f, 0.3f, 0.1f };
	float y[]{ 0.4f, 0.3f, 0.1f }; 
	 
	// Create a neural network model
	NN theNN; 
	
	// Setting the topology of the network.
	// Input variables number: 1 
	// Output variables number: 1 
	// There are 
	//    1 layer 
	//    1 neuro 
	//  in the network.
	theNN.setModel({1,1});

	// Show the topology of the network 
	theNN.show();
	 
	// Testing function 
	//  y = w * x 
	// Trainning the model 
	for (int i = 0; i < 20; i++)
	{ 
		x[0] = 0.1f * i + 0.1f;
		y[0] = 10.0f * x[0];

		theNN.train(x, y, 0.3f);
	}
	 
	// Show the result 
	theNN.show();
	  
	return;
}
 
/**
Test Case Results:
Layer Number: 1

Layer 0
-----------------
   Input: 1
  Output: 1

 Neuro 0
  Weights:
   -2.72593
    
Train:
 Diff:
  -2.54519
Train:
 Diff:
  -5.05983
Train:
 Diff:
  -7.40759
Train:
 Diff:
  -9.34344
Train:
 Diff:
  -10.5581
Train:
 Diff:
  -10.7692
Train:
 Diff:
  -9.85027
Train:
 Diff:
  -7.94776
Train:
 Diff:
  -5.5078
Train:
 Diff:
  -3.14557
Train:
 Diff:
  -1.38405
Train:
 Diff:
  -0.413706
Train:
 Diff:
  -0.0609531
Train:
 Diff:
  0.000919342
Train:
 Diff:
  -0.000175476
Train:
 Diff:
  6.86646e-05
Train:
 Diff:
  -4.19617e-05
Train:
 Diff:
  3.05176e-05
Train:
 Diff:
  -2.67029e-05
Train:
 Diff:
  3.05176e-05

Layer Number: 1

Layer 0
-----------------
   Input: 1
  Output: 1

 Neuro 0
  Weights:
   9.99999
*/
int main(int argc, char* argv[])
{
    std::cout << "Hello World" << std::endl;

	testNN();
	  
	return 1;
}