This a simple Neural Network in C++.

For Windows user, if your working context is Visual Studio 2020, just open the command prompt, then go to '\tmp' folder, 
and trigger 'run.bat'.

A linear regression example.

The target linear model is 

y = 10.0 x 

~~~
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
~~~


The results are like:
~~~
Hello World

Layer Number: 1

Layer 0
-----------------
   Input: 1
  Output: 1

 Neuro 0
  Weights:
   0.586288


Train:
 Diff:
  -1.88274
Train:
 Diff:
  -3.74289
Train:
 Diff:
  -5.47959
Train:
 Diff:
  -6.91159
Train:
 Diff:
  -7.8101
Train:
 Diff:
  -7.9663
Train:
 Diff:
  -7.28651
Train:
 Diff:
  -5.87917
Train:
 Diff:
  -4.07427
Train:
 Diff:
  -2.32686
Train:
 Diff:
  -1.02382
Train:
 Diff:
  -0.306028
Train:
 Diff:
  -0.0450878
Train:
 Diff:
  0.000679016
Train:
 Diff:
  -0.0001297
Train:
 Diff:
  4.95911e-05
Train:
 Diff:
  -2.67029e-05
Train:
 Diff:
  1.90735e-05
Train:
 Diff:
  -1.90735e-05
Train:
 Diff:
  1.90735e-05

Layer Number: 1

Layer 0
-----------------
   Input: 1
  Output: 1

 Neuro 0
  Weights:
   9.99999
~~~
After trainning, the weight we get from the NN model is 9.9999
