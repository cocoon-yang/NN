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
	void clear(); 

	bool isFinish();
	void setFinish(bool state);

public:
	void predict(float* input);
	void train(float* input, float* y, float lr);

	void load(const char* fileName);
	void save(const char* fileName);
	void show(); 
	 
	void killConnection(uint layIndex, uint neuronIndex, uint connectionIndex);
	void activeConnection(uint layIndex, uint neuronIndex, uint connectionIndex);

	void activeRandConnection(uint num);

	void setLayerOrder(uint layIndex, float val);
	void setNeuronOrder(uint layIndex, uint neuronIndex, float val);

	void setWeight(uint layIndex, uint neuronIndex, uint connectionIndex, float val);
private:
	std::vector<Layer*> model;

	std::vector<uint> topology; 

	bool _FINISH;
};

