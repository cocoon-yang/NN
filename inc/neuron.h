#pragma once
#include <iostream>
#include <vector> 
#include <cmath>
#include "ass.h" 

//#define _DEBUG_  1 

//typedef float DataType;
//typedef unsigned int uint;

class Connection;
class Layer;

class Neuron: public std::enable_shared_from_this<Neuron>
{
public:
	Neuron(int num);
	virtual ~Neuron();

	Neuron(const Neuron& RHS);

	Neuron& operator = (const Neuron& RHS);

	std::shared_ptr<Neuron> getPtr();

public:
	void init(Layer* pLayer = nullptr);
	void setWeight(std::vector<float>& values);

	/**
	\brief Calculate the output of the neuron
	@param[in] pVal: input variable vector
	*/
	DataType run();

	/**
	\brief Get the output of the neuron
	*/
	DataType getValue();

	void setValue(DataType val);

	void setOrder(float val);

	int getOrder();

	void setType(uint val);

	std::string toStr();

	uint getConnectionNum();

	/**
	 * @brief Adding a connection to this Neuron 
	 * @param  
	*/
	void connectNeuron(std::shared_ptr<Neuron>);

	/**
	 * @brief Delete the connection to this Neuron
	 * @param
	*/
	void disconnectNeuron(std::shared_ptr<Neuron>);

	/**
	\brief Get one Connection pointer of the neuron
	*/
	std::shared_ptr<Connection> getConnection(size_t index);

	void updataWeight(DataType diffVal, std::shared_ptr<DataType[]> varGrad, DataType learnRate);

	void killConnection(uint connectionIndex);  

	void activeConnection(uint connectionIndex);
private:
	DataType calcuOutput(DataType value);

	DataType calcuGrad(DataType diffVar, DataType value);

public:
	int layerID;
	int index;

private:
	bool _ALIVE;

	uint _type;

	float order;

	DataType _value;
	uint _inputNum;

	std::vector<std::shared_ptr<Connection>> _pConnections;
};

