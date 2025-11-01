#pragma once 
#include "layer.h"

class NeuronNet
{ 
public:
	NeuronNet();
	virtual ~NeuronNet();  

public:
	void setModel(std::vector<uint> theTop);

	void init();

public:
	void predict(float* input);
	void train(float* input, float* y, float lr);
	void save(const char* fileName);

private:
	std::vector<Layer*> model;

	std::vector<uint> topology; 

	bool _FINISH;
};

