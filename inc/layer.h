#pragma once
#include "neuron.h"

class Layer
{
public:
	Layer(int inNum, int outNum, int id);
	virtual ~Layer();

	Layer(const Layer& RHS);

	Layer& operator = (const Layer& RHS);

public:
	void clear();

	int getID();

	int getInputNum();

	int getOutputNum();

	void init(std::shared_ptr<Layer> pLayer = nullptr);

	std::shared_ptr<Neuron> getNeuron(size_t index);

	std::string toStr();
	 
	void show();

	void setOrder(float val);

	void killConnection(uint neuronIndex, uint connectionIndex);
	void activeConnection(uint neuronIndex, uint connectionIndex);

	/**
	 * @brief Active selected connections of the Neuron only.  
	 * @param num: uint, the number of the selected connections.
	*/
	void activeRandConnection(uint num);

public:
	void forward( );
	void backward(std::shared_ptr<DataType[]> output_grad, std::shared_ptr<DataType[]> input_grad, float lr); 
	 
private:
	int _id;

	int _inputNum;
	int _outputNum;
	 
	std::vector<std::shared_ptr<Neuron>> _pNerons;
};

